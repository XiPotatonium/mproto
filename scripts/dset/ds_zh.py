import json
import os
import random
from pathlib import Path
import re
from typing import Dict, List, Optional, Set, Tuple, Union
from collections import Counter
import numpy as np
from rich.console import Console
from rich.progress import Progress

import typer
from loguru import logger
from stanza.server import CoreNLPClient
from alchemy.util.extention.rich import full_columns

from . import filter_overlapping, filter_partial_overlapping
from ..trie import TrieChar
from .. import prepare_logging, zh_join
from ..stanza import ChunkResult, MyParseTree, StanzaHelper, Token


class DictAugNlpHelper(StanzaHelper):
    def __init__(
        self,
        client: CoreNLPClient,
        preserve_trie: Optional[TrieChar], join_vocab: set,
        preserve_case_sensitive: bool,
        sep: str,
    ):
        super().__init__(client)
        self.preserve_trie = preserve_trie
        self.join_vocab = join_vocab
        self.sep = sep
        self.counter = Counter()
        self.preserve_case_sensitive = preserve_case_sensitive

    def chunk(self, text: str) -> List[ChunkResult]:
        ret = []

        doc = self.client.annotate(text)
        for sentence in doc.sentence:
            tokens = sentence.token
            tokens: List[Token] = [Token.from_stanza(t) for t in tokens]

            constituent_tree = sentence.parseTree
            constituent_tree = StanzaHelper._annotate_tree_span(constituent_tree)
            assert constituent_tree.yieldEndIndex == len(tokens)

            constituent_tree, joined_tokens =  self._join(constituent_tree, tokens)

            nps = []  # (token start, token end)
            StanzaHelper._find_np(constituent_tree, tokens, nps)

            ret.append(ChunkResult(tokens=joined_tokens, nps=nps))
        return ret

    def _join(self, root: MyParseTree, tokens: List[Token]) -> Tuple[MyParseTree, List[str]]:
        nodes_to_join: List[MyParseTree] = []
        self.__join(root, tokens, nodes_to_join)

        join_tags = np.ones(len(tokens), dtype=int) * -1
        for join_result_i, join_result in enumerate(nodes_to_join):
            join_tags[join_result.yieldBeginIndex] = join_result_i
            if join_result.yieldBeginIndex + 1 < join_result.yieldEndIndex:
                # -2相当于tagging的I
                join_tags[join_result.yieldBeginIndex + 1: join_result.yieldEndIndex] = -2

        joined_tokens = []
        for tok, tok_join_tag in zip(tokens, join_tags):
            if tok_join_tag == -1:
                joined_tokens.append(tok)
            elif tok_join_tag >= 0:
                sub_tree = nodes_to_join[tok_join_tag]
                sub_tree_tokens = [t.word for t in tokens[sub_tree.yieldBeginIndex: sub_tree.yieldEndIndex]]
                joined_tok = Token(
                    word=self.sep.join(sub_tree_tokens),
                    pos=sub_tree.value      # 用成分标记代替POS
                )
                joined_tokens.append(joined_tok)
                self.counter.update([joined_tok.word])

                logger.debug("Join tokens {}".format(sub_tree_tokens))

        self.__merge_tree(root, set(map(lambda x: id(x), nodes_to_join)))
        joined_tree = self._annotate_tree_span(root)
        assert joined_tree.yieldEndIndex == len(joined_tokens)

        return joined_tree, joined_tokens

    @staticmethod
    def __merge_tree(tree: MyParseTree, joined_tree_ids: Set[int]):
        if id(tree) in joined_tree_ids:
            tree.child = [tree.child[0]]
            tree.child[0].child = []        # tree.child[0]将会装作是原来的token，只是value并不对
            return

        for child in tree.child:
            DictAugNlpHelper.__merge_tree(child, joined_tree_ids)

    def __join(self, tree: MyParseTree, tokens: List[Token], join_results: List[MyParseTree]):
        """

        Args:
            tree (ParseTree): _description_
            tokens (List[Token]): _description_
            join_results (List[ParseTree]): _description_
        """
        if tree.yieldBeginIndex + 1 == tree.yieldEndIndex:
            # 只join词组，不用管单个词
            return
        children_are_all_token = True
        for child in tree.child:
            if child.yieldBeginIndex + 1 < child.yieldEndIndex:
                children_are_all_token = False
                break
        if children_are_all_token:
            tree_phrase = self.sep.join([t.word for t in tokens[tree.yieldBeginIndex: tree.yieldEndIndex]])
            if tree_phrase in self.join_vocab:
                # 是一个高度为1的节点，并且自己在词表中
                has_preserve_item = False
                if self.preserve_trie is not None:
                    for t_i in range(len(tree_phrase)):
                        matches = self.preserve_trie.startswith(tree_phrase, t_i, case_senitive=self.preserve_case_sensitive)
                        if (
                            len(matches) != 0 and
                            not (t_i == 0 and matches[-1] == len(tree_phrase))
                        ):
                            # 如果存在匹配，且不是完全匹配（fullmatch)，那意味着这个词有一部分是种子词，如果join了会丢失远程监督匹配
                            has_preserve_item = True
                            break
                if not has_preserve_item:
                    # 我们需要防止种子词被切碎
                    join_results.append(tree)
        else:
            for child in tree.child:
                self.__join(child, tokens, join_results)


app = typer.Typer()


@app.command("ds")
def distant_annotate_entry(
    seed_path: str,
    output_dir: str = typer.Option(...),
    corpus_dir: str = "data/corpus/weixin-antif",
    char_level: bool = False,
    nested: bool = False,
    log_level: str = "INFO",
    port: str = "9007",
    dev: bool = True,
):
    """

    Args:
        seed_path (str): _description_
        output_dir (str): _description_
        corpus_dir (str): _description_
        char_level (bool): 分不分词，如果不分词，那么join_vocab以及additional_preserve_vocab不起效
        join_vocab (Optional[str]): 用这个词典合并切词结果
        additional_preserve_vocab (Optional[str]): 除了seeds之外还需要preserve哪些
        nested (bool, optional): 嵌套NER还是Flat NER. Default False
        log_level (str, optional): 日志level. default "INFO"
    """
    seed_path: Path = Path(seed_path)
    with seed_path.open('r', encoding="utf8") as f:
        seeds = json.load(f)
    if dev:
        with seed_path.with_stem(seed_path.stem + "_dev").open('r', encoding="utf8") as f:
            dev_only_seeds = json.load(f)
    else:
        dev_only_seeds = {}
    console = prepare_logging(level=log_level)
    distant_annotate(
        seeds=seeds,
        dev_only_seeds=dev_only_seeds,
        output_dir=output_dir,
        corpus_dir=corpus_dir,
        char_level=char_level,
        nested=nested,
        console=console,
        log_level=log_level,
        port=port
    )


def distant_annotate(
    seeds: Dict[str, List[str]],
    dev_only_seeds: Dict[str, List[str]],
    output_dir: str,
    corpus_dir: str = "data/corpus/weixin-antif",
    char_level: bool = False,
    nested: bool = False,
    console: Optional[Console] = None,
    log_level: str = "INFO",
    port: str = "9007",
):
    """为了保证recall，这里还是采取token level的occur

    Args:
        seeds (Dict[str, List[str]]): _description_
        dev_only_seeds (Dict[str, List[str]]): _description_
        output_dir (str): _description_
        corpus_dir (str): _description_
        char_level (bool): 分不分词，如果不分词，那么join_vocab以及additional_preserve_vocab不起效
        nested (bool, optional): 嵌套NER还是Flat NER. Default False
    """
    if console is None:
        console = prepare_logging(log_level)

    flat_dev_only_seeds = set()
    for ty_dev_only_seeds in dev_only_seeds.values():
        flat_dev_only_seeds.update(map(lambda x: x.lower(), ty_dev_only_seeds))

    seed_tries = {}
    seed2ty = {}
    for ty, vs in seeds.items():
        seed_trie = TrieChar()
        for seed in vs:
            assert seed not in seed2ty, "{} has multiple tag {} and {}".format(seed, ty, seed2ty[seed])
            seed2ty[seed] = ty
            seed_trie.add_word(seed, case_sensitive=False)
        seed_tries[ty] = seed_trie
    logger.info("Seeds statistic {}, {} in total".format({ty: len(vs) for ty, vs in seeds.items()}, len(seed2ty)))

    corpus_dir: Path = Path(corpus_dir)
    samples = []
    train_samples = []
    dev_samples = []
    counters = {ty: 0 for ty in seeds.keys()}
    os.environ["NO_PROXY"] = "localhost"
    with CoreNLPClient(
            threads=8, timeout=30000, memory='16G', endpoint="http://localhost:{}".format(port),
            annotators=['tokenize', 'ssplit', 'pos', 'parse'], properties="chinese", be_quiet=True
    ) as client, \
            Progress(*full_columns(), console=console) as progress:
        nlp = None
        if not char_level:
            nlp = StanzaHelper(client)

        files: List[Path] = list(corpus_dir.iterdir())
        files.sort()
        tid = progress.add_task("", total=len(files))
        idx = 0
        for file in files:
            with file.open("r", encoding="utf8") as f:
                article = json.load(f)

            all_pos = []
            all_tok = []
            for section in article["sections"]:
                if section == "往期精彩回顾":
                    # logger.debug("Early stop at \"{}\" (p {}/{})".format(raw_sent, p_i, len(sents)))
                    break

                if nlp is not None:
                    # 用additional vocab来聚合切的过碎的词，但同时会preserve种子词
                    join_results = nlp.chunk(section)
                    for join_result in join_results:
                        all_tok.append([t.word for t in join_result.tokens])
                        all_pos.append([t.pos for t in join_result.tokens])
                else:
                    all_tok.append(list(section))
                    all_pos.append(None)

            for s_i, (tok, pos) in enumerate(zip(all_tok, all_pos)):
                # NOTE: 这里使用和zh_join类似的逻辑，因为直接用''进行join，可能会把两个英文粘到一起
                tok2char = {}
                char2tok = {}

                puct_rule = "\·\~\！\@\#\￥\%\……\&\*\（\）\——\-\+\=\【\】\{\}\、\|\；\‘\’\：\“\”\《\》\？" \
                            "\，\。\、\`\~\!\#\$\%\^\&\*\(\)\_\[\]{\}\\\|\;\'\'\:\"\"\,\.\/\<\>\?"
                zh_rule = "\u4e00-\u9fa5"
                rule = f"[{puct_rule}{zh_rule}]"

                sent = ''
                for i, t in enumerate(tok):
                    tok2char[i] = len(sent)
                    char2tok[len(sent)] = i
                    sent += t
                    # 中文和标点符号不需要空格，因此两侧都不是中文和标点的情况下需要添加空格
                    if (
                        not re.fullmatch(rule, t[-1]) and
                        (i + 1 < len(tok) and not re.fullmatch(rule, tok[i + 1][0]))
                    ):
                        sent += ' '

                # 在远程监督匹配的时候可能需要忽视字母的大小写来满足更多匹配
                matches = []
                for tok_i in range(len(tok)):
                    for ty, trie in seed_tries.items():
                        matches_ = trie.startswith(sent, tok2char[tok_i], case_senitive=False)
                        for match_ in matches_:
                            if match_ in char2tok:
                                match_ = char2tok[match_]
                            else:
                                # 不是一个完整的token
                                continue
                            # 因为seed不能有两个tag，所以这里不会有冲突
                            matches.append({"start": tok_i, "end": match_, "type": ty})
                # 按长度进行过滤
                matches = filter_partial_overlapping(matches)
                if not nested:
                    matches = filter_overlapping(matches)

                for e in matches:
                    counters[e["type"]] += 1
                sample = {
                    "tokens": tok,
                    "entities": matches,
                    "ltokens": [] if s_i == 0 else all_tok[s_i - 1],
                    "rtokens": [] if s_i == len(all_tok) - 1 else all_tok[s_i + 1],
                    "id": str(idx),
                    "doc_id": file.name,
                }
                if pos is not None:
                    sample["pos"] = pos
                idx += 1
                samples.append(sample)
                if len(matches) != 0:
                    dev_only_hit = False
                    for m in matches:
                        phrase = zh_join(tok[m["start"]:m["end"]])
                        if phrase.lower() in flat_dev_only_seeds:
                            dev_only_hit = True
                    if dev_only_hit:
                        dev_samples.append(sample)
                    else:
                        train_samples.append(sample)

            progress.advance(tid)

    logger.info("{} sents. {}".format(len(samples), counters))
    logger.info("Split into {} train samples and {} dev samples".format(len(train_samples), len(dev_samples)))
    meta = {
        "seeds": seeds,
        "dev_only_seeds": dev_only_seeds,
        "entities": counters,
    }

    output_dir: Path = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "train.jsonl").open("w", encoding="utf8") as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    with (output_dir / "dev.jsonl").open("w", encoding="utf8") as f:
        for sample in dev_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    with (output_dir / "meta.json").open("w", encoding="utf8") as f:
        meta["entities"] = {
            ty: {"short": ty, "verbose": ty, "count": count} for ty, count in meta["entities"].items()
        }
        json.dump(meta, f, ensure_ascii=False)

    for sample in samples:
        sample["entities"] = []     # inference不需要entities，防止冲突
    with (output_dir / "inference.jsonl").open("w", encoding="utf8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')


@app.command()
def sample_seeds(seeds_path: str, num_dev_only_seeds: int = 75):
    seeds_path: Path = Path(seeds_path)
    with seeds_path.open('r', encoding="utf8") as f:
        seeds = json.load(f)
    dev_only_seeds = {}
    for ty, ty_seeds in seeds.items():
        dev_only_seeds[ty] = random.sample(ty_seeds, num_dev_only_seeds)

    print("All seeds count:")
    for ty, ty_seeds in seeds.items():
        print(f"{ty}: {len(ty_seeds)}")

    print("Dev seeds count:")
    for ty, ty_seeds in dev_only_seeds.items():
        print(f"{ty}: {len(ty_seeds)}")

    with seeds_path.with_stem(seeds_path.stem + "_dev").open('w', encoding="utf8") as f:
        json.dump(dev_only_seeds, f, ensure_ascii=False)


if __name__ == '__main__':
    app()