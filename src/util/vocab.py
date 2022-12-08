from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import gc
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import warnings
from loguru import logger
import numpy as np



@dataclass
class Vocab:
    idx2word: List[str]
    word2idx: Dict[str, int]

    def __len__(self):
        return len(self.idx2word)


def strip_vocab(
    w2v: Optional[Any],
    words: Iterable[str],
    special_tokens: List[str],
    threshold=-1,
    nwords=-1,
):
    idx2word = []
    # Construct word2vec with corpus
    counter = Counter(words)
    if nwords >= 0:
        # TODO: 用Counter的most_common选取nwords
        raise NotImplementedError()
    else:
        for k, v in counter.items():
            if v >= threshold and (w2v is None or k in w2v):
                idx2word.append(k)
    idx2word.sort()         # 为了保证多次从corpus建立的vocab是一致的，这里对vocab进行排序
    idx2word = special_tokens + idx2word        # 要保证特殊token的顺序不变
    word2idx = {w: i for i, w in enumerate(idx2word)}
    return Vocab(idx2word=idx2word, word2idx=word2idx)

    
def build_dictionary(
    filename: str,
    corpus_paths: Optional[List[str]],
    cache_dir: str,
    cache_tag: Optional[str],
    special_tokens: List[str],
    threshold=-1,
    nwords=-1,
    padding_factor=8,
    tolower=True,
    ignore_missing_ebd_cache=False,
) -> Tuple[Vocab, Path, Optional[np.ndarray], Path]:
    cache_dir: Path = Path(cache_dir)
    w2v_path: Path = Path(filename)
    # 用with suffix会有bug
    w2v_cache_json_path = cache_dir / (
        (w2v_path.stem if cache_tag is None else "{}_{}".format(w2v_path.stem, cache_tag))  + ".json"
    )
    w2v_cache_npy_path = cache_dir / (
        (w2v_path.stem if cache_tag is None else "{}_{}".format(w2v_path.stem, cache_tag))  + ".npy"
    )
    if corpus_paths is not None:
        corpus_paths = [Path(p) for p in corpus_paths]

    if (
        corpus_paths is None or (
            w2v_cache_json_path.exists() and w2v_cache_npy_path.exists() and
            min(w2v_cache_json_path.stat().st_mtime, w2v_cache_npy_path.stat().st_mtime) >
            max([p.stat().st_mtime for p in corpus_paths] + [w2v_path.stat().st_mtime])          
        )
    ):
        # 缓存存在，且缓存的最旧时间新于语料和词嵌入的最新时间，那么缓存有效，或者未指定语料，只能依赖缓存
        if corpus_paths is None:
            logger.warning("Corpus not specified, force using cache, there might be some errors")
        logger.info(
            "Reuse vocab and word embedding from \"{}\" and \"{}\"",
            w2v_cache_json_path, w2v_cache_npy_path
        )
        with w2v_cache_json_path.open('r', encoding="utf8") as f:
            idx2word = json.load(f)
            word2idx = {w: i for i, w in enumerate(idx2word)}
            vocab = Vocab(idx2word=idx2word, word2idx=word2idx)
        if w2v_cache_npy_path.exists():
            with w2v_cache_npy_path.open('rb') as f:
                ebd = np.load(f)
        elif ignore_missing_ebd_cache:
            logger.warning(
                f"Embedding cache {w2v_cache_npy_path} not exists. "
                "It is only expected when using trained model"
            )
            ebd = None
        else:
            raise Exception(f"Embedding cache {w2v_cache_npy_path} not exists")
    else:
        warnings.filterwarnings("ignore")
        from gensim.models import KeyedVectors
        from gensim.scripts.glove2word2vec import glove2word2vec
        logger.info("Constructing vocab...")
        if w2v_path.suffix == ".txt":
            if "w2v" in w2v_path.name:
                logger.info("Load raw word2vec from \"{}\", assume word2vec format".format(w2v_path))
                word2vec = KeyedVectors.load_word2vec_format(str(w2v_path), binary=False)
            else:
                logger.info("Load raw word2vec from \"{}\", assume glove format".format(w2v_path))
                # Path.with_stem not available under python 3.9
                w2v_output_path = w2v_path.with_name(w2v_path.stem + ".w2v" + w2v_path.suffix)
                count, dims = glove2word2vec(str(w2v_path), str(w2v_output_path))
                word2vec = KeyedVectors.load_word2vec_format(str(w2v_output_path), binary=False)
        elif w2v_path.suffix == ".kv":
            logger.info("Load raw word2vec from \"{}\"".format(w2v_path))
            word2vec = KeyedVectors.load(str(w2v_path))
        else:
            raise ValueError("Unknown word2vec file format (\"{}\"). Expect .txt or .kv".format(w2v_path))

        def tokens():
            for corpus_path in corpus_paths:
                with corpus_path.open('r', encoding="utf8") as f:
                    if corpus_path.suffix == ".json":
                        documents = json.load(f)
                    elif corpus_path.suffix == ".jsonl":
                        documents = map(lambda l: json.loads(l), f)
                    else:
                        raise ValueError("Unknown data format {}, expect .json or .jsonl".format(corpus_path))
                    for doc in documents:
                        for tok in doc["tokens"]:
                            yield tok.lower() if tolower else tok

        vocab = strip_vocab(word2vec, tokens(), special_tokens, threshold, nwords)

        # TODO: padding
        ebd = np.zeros((len(vocab), len(word2vec[word2vec.index_to_key[0]])), dtype=float)
        for idx, word in enumerate(vocab.idx2word):
            if word in word2vec:
                ebd[idx, :] = deepcopy(word2vec[word])
        # 手动GC，因为可能比较大
        del word2vec
        gc.collect()

        # 保存cache
        logger.info(
            "Dump vocab and embedding cache at \"{}\" and \"{}\"",
            w2v_cache_json_path, w2v_cache_npy_path
        )
        with w2v_cache_json_path.open('w', encoding="utf8") as f:
            json.dump(vocab.idx2word, f, ensure_ascii=False)
        with w2v_cache_npy_path.open('wb') as f:
            np.save(f, ebd)

    return vocab, w2v_cache_json_path, ebd, w2v_cache_npy_path


def build_pos_dictionary(
    corpus_paths: Optional[List[str]],
    cache_dir: str,
    cache_tag: Optional[str],
    special_tokens: List[str],
    threshold=-1,
    nwords=-1,
    padding_factor=8,
) -> Tuple[Vocab, Path]:
    cache_dir = Path(cache_dir)
    # 用with suffix会有bug
    w2v_cache_json_path = cache_dir / (
        ("pos" if cache_tag is None else "pos_{}".format(cache_tag))  + ".json"
    )
    if corpus_paths is not None:
        corpus_paths = [Path(p) for p in corpus_paths]

    if (
        corpus_paths is None or (
            w2v_cache_json_path.exists() and
            w2v_cache_json_path.stat().st_mtime >
            max(p.stat().st_mtime for p in corpus_paths)
        )
    ):
        # 缓存存在，且缓存的最旧时间新于语料和词嵌入的最新时间，那么缓存有效，或者未指定语料，只能依赖缓存
        if corpus_paths is None:
            logger.warning("Corpus not specified, force using cache, there might be some errors")
        logger.info("Reuse vocab and from \"{}\"", w2v_cache_json_path)
        with w2v_cache_json_path.open('r', encoding="utf8") as f:
            idx2word = json.load(f)
            word2idx = {w: i for i, w in enumerate(idx2word)}
            vocab = Vocab(idx2word=idx2word, word2idx=word2idx)
    else:
        def tags():
            for corpus_path in corpus_paths:
                with corpus_path.open('r', encoding="utf8") as f:
                    if corpus_path.suffix == ".json":
                        documents = json.load(f)
                    elif corpus_path.suffix == ".jsonl":
                        documents = map(lambda l: json.loads(l), f)
                    else:
                        raise ValueError("Unknown data format {}, expect .json or .jsonl".format(corpus_path))
                    for doc in documents:
                        for tag in doc.get("pos", []):
                            yield tag

        vocab = strip_vocab(None, tags(), special_tokens, threshold, nwords)

        # 保存cache
        logger.info("Dump vocab and embedding cache at \"{}\"".format(w2v_cache_json_path))
        with w2v_cache_json_path.open('w', encoding="utf8") as f:
            json.dump(vocab.idx2word, f, ensure_ascii=False)

    return vocab, w2v_cache_json_path