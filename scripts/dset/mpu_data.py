"""
为了处理https://github.com/kangISU/Conf-MPU-DS-NER的数据
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger
import numpy as np
import copy

import typer
from rich.progress import track
from ..trie import Trie

from . import filter_overlapping, filter_partial_overlapping, prfs
from .. import prepare_logging


def lookup_in_Dic(tag2idx: Dict[str, int], dicFile: Path, sentences: List, tag: str, windowSize: int):
    tagIdx = tag2idx[tag]
    dic = set()
    labeled_word = set()
    count = 0
    with dicFile.open("r", encoding='utf8') as fw:
        for line in fw:
            line = line.strip()
            if len(line) > 0:
                dic.add(line)
    for i, sentence in enumerate(track(sentences)):
        wordList = [word for word, label, dicFlags in sentence]
        trueLabelList = [label for word, label, dicFlags in sentence]
        isFlag = np.zeros(len(trueLabelList))
        j = 0
        # 这就是最长贪婪匹配吧
        while j < len(wordList):
            Len = min(windowSize, len(wordList) - j)
            k = Len
            while k >= 1:
                words = wordList[j:j + k]
                words_ = " ".join([w for w in words])

                if words_ in dic:
                    isFlag[j:j + k] = 1
                    j = j + k
                    break
                k -= 1
            j += 1

        for m, flag in enumerate(isFlag):
            if flag == 1:
                count += 1
                labeled_word.add(sentence[m][0])
                sentence[m][2][tagIdx] = 1
                # print(sentence)

    return sentences, len(labeled_word), count


def decode_tags(tags: List[int], idx2tag: List[str]) -> List[Dict[str, Any]]:
    ret = []
    last_b_idx = -1
    for i, tag in enumerate(tags):
        if tag == 0:
            # O
            if last_b_idx >= 0:
                # 一个span结尾
                ret.append({"start": last_b_idx, "end": i, "type": idx2tag[tags[last_b_idx] - 1]})
                last_b_idx = -1
        else:
            # I tag
            if last_b_idx < 0:
                # begin
                last_b_idx = i
    if last_b_idx >= 0:
        ret.append({"start": last_b_idx, "end": len(tags), "type": idx2tag[tags[last_b_idx] - 1]})

    return ret


def distant_anno(dict_files: Dict[str, Path], file: Path, tag2idx: Dict[str, int]):
    tag2idx_copy = copy.deepcopy(tag2idx)
    tag2idx_copy.pop('O')
    new_tag2idx = {}
    idx = 0
    for tag in tag2idx_copy.keys():
        new_tag2idx[tag] = idx
        idx += 1

    classNum = len(new_tag2idx)
    sentences = []
    with file.open("r", encoding='utf8') as fw:
        sentence = []
        for line in fw:
            if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == '\n':
                if len(sentence) > 0:
                    sentences.append(sentence)
                    sentence = []
                continue
            else:
                splits = line.split(' ')
                sentence.append([splits[0].strip(), splits[1].strip(), np.zeros(classNum)])

        if len(sentence) > 0:
            sentences.append(sentence)

    maxLen = 10         # 为什么要限制最大长度？
    for tag, dict_file in dict_files.items():
        logger.info("Matching {}".format(tag))
        sentences, num, count = lookup_in_Dic(new_tag2idx, dict_file, sentences, tag, maxLen)

    logger.info("Matching using my greedy matcher")
    trie = Trie()
    phrase2type = {}
    for tag, dict_file in dict_files.items():
        with dict_file.open("r", encoding='utf8') as fw:
            for line in fw:
                line = line.strip()
                if len(line) > 0:
                    if line in phrase2type:
                        logger.warning("Ambiguous {} in dict, set to O type".format(line))
                        phrase2type[line] = "O"
                    else:
                        phrase2type[line] = tag
    phrase2type = {k: v for k, v in phrase2type.items() if v != "O"}        # 去掉O
    for phrase in phrase2type.keys():
        trie.add_word(phrase.split(), case_sensitive=True)

    samples = []
    for sentence in track(sentences):
        tokens = [word for word, label, dicFlags in sentence]
        fully_labels = [label for word, label, dicFlags in sentence]
        mpu_labels = [
            int(dicFlags.argmax()) + 1 if dicFlags.sum() == 1 else 0
            for word, label, dicFlags in sentence
        ]
        matches = []
        for i in range(len(tokens)):
            matches_ = trie.startswith(tokens[i: min(i + maxLen, len(tokens))], case_senitive=True)
            for l in matches_:
                end = i + l
                matches.append({"type": phrase2type[' '.join(tokens[i: end])], "start": i, "end": end})
        # 按长度进行过滤
        matches = filter_partial_overlapping(matches)
        matches = filter_overlapping(matches)

        # 将fully_labels转换为gt_entities
        fully_entities = []
        last_b_idx = -1
        for i, tag in enumerate(fully_labels):
            if tag.startswith("I"):
                pass
            elif tag == "O":
                if last_b_idx >= 0:
                    # 一个span结尾
                    fully_entities.append({"start": last_b_idx, "end": i, "type": fully_labels[last_b_idx][2:]})
                    last_b_idx = -1
            else:
                if last_b_idx >= 0:  # 一个span结尾
                    fully_entities.append({"start": last_b_idx, "end": i, "type": fully_labels[last_b_idx][2:]})
                    last_b_idx = -1
                last_b_idx = i
        if last_b_idx >= 0:
            fully_entities.append({"start": last_b_idx, "end": len(fully_labels), "type": fully_labels[last_b_idx][2:]})

        labels = np.zeros(len(tokens), dtype=int)
        for match in matches:
            labels[match["start"]: match["end"]] = new_tag2idx[match["type"]] + 1
        sample = {
            "tokens": tokens,
            "fully_labels": fully_labels,
            "mpu_labels": mpu_labels,
            "ds_labels": labels,
            "ds_entities": matches,
            "fully_entities": fully_entities,
        }
        samples.append(sample)
        # 对拍，保证生成的正确性
        # 注意MPU的生成其实和我们的匹配是不一样的，并不能保证一致性
        # MPU和我们都是首先做最长贪婪匹配，但是MPU对每一个类别的词典独立进行匹配，
        # 如果发现某个token被两个类别的匹配覆盖到了，就认为有歧义，然后将这个token标为O
        # 而我们从词典层面消歧，如果一个词属于多个类别，那么将这个词从每个词典中剔除
        # MPU这种token式的消歧方式肯定不适合span-based方法
        # assert len(labels) == len(mpu_labels) and all(label == their_label for label, their_label in zip(labels, mpu_labels)), sample

    return samples


def copy_anno(file: Path, tag2idx: Dict[str, int]):
    idx2tag = {v: k for k, v in tag2idx.items()}

    classNum=len(tag2idx) - 1
    sentences = []
    with file.open("r", encoding='utf8') as fw:
        sentence = []
        for line in fw:
            if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == '\n':
                if len(sentence) > 0:
                    sentences.append(sentence)
                    sentence = []
                continue
            else:
                splits = line.split(' ')
                sentence.append([splits[0].strip(), splits[1].strip(), int(splits[2].strip())])

        if len(sentence) > 0:
            sentences.append(sentence)

    samples = []
    for sentence in track(sentences):
        tokens = [word for word, label, flag in sentence]
        fully_labels = [label for word, label, flag in sentence]
        mpu_labels = [flag for word, label, flag in sentence]

        # 将fully_labels转换为gt_entities
        fully_entities = []
        last_b_idx = -1
        for i, tag in enumerate(fully_labels):
            if tag.startswith("I"):
                pass
            elif tag == "O":
                if last_b_idx >= 0:
                    # 一个span结尾
                    fully_entities.append({"start": last_b_idx, "end": i, "type": fully_labels[last_b_idx][2:]})
                    last_b_idx = -1
            else:
                if last_b_idx >= 0:  # 一个span结尾
                    fully_entities.append({"start": last_b_idx, "end": i, "type": fully_labels[last_b_idx][2:]})
                    last_b_idx = -1
                last_b_idx = i
        if last_b_idx >= 0:
            fully_entities.append({"start": last_b_idx, "end": len(fully_labels), "type": fully_labels[last_b_idx][2:]})

        # 将mpu的标注转换为ds_entities
        ds_entities = []
        start_idx = -1
        for i, tag in enumerate(mpu_labels):
            if i != 0 and tag == mpu_labels[i - 1]:
                # 和前一个相同 I
                pass
            elif tag == tag2idx["O"]:
                if start_idx >= 0:
                    # 一个span结尾
                    ds_entities.append({"start": start_idx, "end": i, "type": idx2tag[mpu_labels[start_idx]]})
                    start_idx = -1
            else:
                # (i == 0 or tag和前一个不同) and tag不是O
                if start_idx >= 0:
                    # 一个span结尾
                    ds_entities.append({"start": start_idx, "end": i, "type": idx2tag[mpu_labels[start_idx]]})
                    start_idx = -1
                start_idx = i
        if start_idx >= 0:
            ds_entities.append({"start": start_idx, "end": len(mpu_labels), "type": idx2tag[mpu_labels[start_idx]]})

        sample = {
            "tokens": tokens,
            "fully_labels": fully_labels,
            "ds_labels": mpu_labels,
            "ds_entities": ds_entities,
            "fully_entities": fully_entities,
        }
        samples.append(sample)

    return samples


app = typer.Typer()


def __ds(
    dataset_dir: Path,
    output_dir: Path,
    dict_paths: Dict[str, Path],
    tag2idx: Dict[str, int],
):
    console = prepare_logging()

    output_dir.mkdir(exist_ok=True, parents=True)

    trainFile = dataset_dir / 'train.txt'
    logger.info("Load train samples from \"{}\"".format(trainFile))
    train_samples = distant_anno(dict_paths, trainFile, tag2idx)
    # train_samples = tagging2span(train_samples, tag2idx)

    """
    train f1，原论文算的是token-level的precision，我们是span-based方法，token-level precision没有意义，所以算span的precision和recall
    """
    # matching_f1(train_samples, tag2idx)
    label_mapper = lambda es: [(e["start"], e["end"], e["type"]) for e in es]
    prfs(
        [label_mapper(sample["fully_entities"]) for sample in train_samples],
        [label_mapper(sample["ds_entities"]) for sample in train_samples],
        console=console,
    )

    # 训练数据选择远程监督的结果作为gt
    with (output_dir / "train.jsonl").open('w', encoding="utf8") as wf:
        for i, sample in enumerate(train_samples):
            train_sample = {
                "tokens": sample["tokens"],
                "entities": sample["ds_entities"],
                "human_entities": sample["fully_entities"],
                "id": "train{}".format(i)
            }
            wf.write(json.dumps(train_sample, ensure_ascii=False))
            wf.write('\n')

    val_file = dataset_dir / 'valid.txt'
    logger.info("Load val samples from \"{}\"".format(val_file))
    val_samples = distant_anno(dict_paths, val_file, tag2idx)

    test_file = dataset_dir / 'test.txt'
    logger.info("Load test samples from \"{}\"".format(test_file))
    test_samples = distant_anno(dict_paths, test_file, tag2idx)

    # 存储原始的test和val，因为要用精确的标注来评估
    def dev_converter(sample: Dict[str, Any]):
        return {
            "tokens": sample["tokens"],
            "entities": sample["fully_entities"],
        }

    with (output_dir / "dev.jsonl").open('w', encoding="utf8") as wf:
        for sample in val_samples:
            wf.write(json.dumps(dev_converter(sample), ensure_ascii=False))
            wf.write('\n')
    with (output_dir / "test.jsonl").open('w', encoding="utf8") as wf:
        for sample in test_samples:
            wf.write(json.dumps(dev_converter(sample), ensure_ascii=False))
            wf.write('\n')

    meta = {
        "entities": {ty: {"short": ty, "verbose": ty} for ty in dict_paths.keys()},
        "type2idx": tag2idx,
    }
    with (output_dir / "meta.json").open('w', encoding="utf8") as wf:
        json.dump(meta, wf, ensure_ascii=False)

    seeds = {}
    for ty, path in dict_paths.items():
        ty_seeds = set()
        with path.open('r', encoding="utf8") as rf:
            for line in rf:
                ty_seeds.add(line.strip())
        seeds[ty] = list(ty_seeds)
    with (output_dir / "seeds.json").open('w', encoding="utf8") as wf:
        json.dump(seeds, wf, ensure_ascii=False)


def __copy(
    dataset_dir: Path,
    output_dir: Path,
    dict_paths: Dict[str, Path],
    tag2idx: Dict[str, int],
):
    console = prepare_logging()

    output_dir.mkdir(exist_ok=True, parents=True)

    trainFile = dataset_dir / 'train.ALL.txt'
    logger.info("Load train samples from \"{}\"".format(trainFile))
    train_samples = copy_anno(trainFile, tag2idx)
    # train_samples = tagging2span(train_samples, tag2idx)

    """
    train f1，原论文算的是token-level的precision，我们是span-based方法，token-level precision没有意义，所以算span的precision和recall
    """
    # matching_f1(train_samples, tag2idx)
    label_mapper = lambda es: [(e["start"], e["end"], e["type"]) for e in es]
    prfs(
        [label_mapper(sample["fully_entities"]) for sample in train_samples],
        [label_mapper(sample["ds_entities"]) for sample in train_samples],
        console=console,
    )

    # 训练数据选择远程监督的结果作为gt
    with (output_dir / "train.jsonl").open('w', encoding="utf8") as wf:
        for i, sample in enumerate(train_samples):
            train_sample = {
                "tokens": sample["tokens"],
                "entities": sample["ds_entities"],
                "human_entities": sample["fully_entities"],
                "id": "train{}".format(i)
            }
            wf.write(json.dumps(train_sample, ensure_ascii=False))
            wf.write('\n')

    val_file = dataset_dir / 'valid.txt'
    logger.info("Load val samples from \"{}\"".format(val_file))
    val_samples = copy_anno(val_file, tag2idx)

    test_file = dataset_dir / 'test.txt'
    logger.info("Load test samples from \"{}\"".format(test_file))
    test_samples = copy_anno(test_file, tag2idx)

    # 存储原始的test和val，因为要用精确的标注来评估
    def dev_converter(sample: Dict[str, Any]):
        return {
            "tokens": sample["tokens"],
            "entities": sample["fully_entities"],
        }

    with (output_dir / "dev.jsonl").open('w', encoding="utf8") as wf:
        for sample in val_samples:
            wf.write(json.dumps(dev_converter(sample), ensure_ascii=False))
            wf.write('\n')
    with (output_dir / "test.jsonl").open('w', encoding="utf8") as wf:
        for sample in test_samples:
            wf.write(json.dumps(dev_converter(sample), ensure_ascii=False))
            wf.write('\n')

    meta = {
        "entities": {ty: {"short": ty, "verbose": ty} for ty in dict_paths.keys()},
        "type2idx": tag2idx,
    }
    with (output_dir / "meta.json").open('w', encoding="utf8") as wf:
        json.dump(meta, wf, ensure_ascii=False)



@app.command()
def conll03(
    dataset_dir: str,
    output_dir: str = typer.Option(...),
    dict_dir: str = typer.Option(...),
    copy: bool = False,
):
    dict_dir: Path = Path(dict_dir)
    dict_paths = {
        "PER": dict_dir / "person.txt",
        "ORG": dict_dir / "organization.txt",
        "LOC": dict_dir / "location.txt",
        "MISC": dict_dir / "misc.txt",
    }

    tag2idx = {"O": 0, "PER": 1, "LOC": 2, "ORG": 3, "MISC": 4}
    if copy:
        __copy(Path(dataset_dir), Path(output_dir), dict_paths, tag2idx)
    else:
        __ds(Path(dataset_dir), Path(output_dir), dict_paths, tag2idx)


@app.command()
def bc5cdr(
    dataset_dir: str,
    output_dir: str = typer.Option(...),
    dict_dir: str = typer.Option(...),
    copy: bool = False,
):
    dict_dir: Path = Path(dict_dir)
    dict_paths = {
        "Chemical": dict_dir / "Chemical.txt",
        "Disease": dict_dir / "Disease.txt"
    }

    tag2idx = {"O": 0, "Chemical": 1, "Disease": 2}
    if copy:
        __copy(Path(dataset_dir), Path(output_dir), dict_paths, tag2idx)
    else:
        __ds(Path(dataset_dir), Path(output_dir), dict_paths, tag2idx)


@app.command()
def dic_eq(a: str, b: str):
    def read_dic(path: Path):
        ret = set()
        with path.open('r', encoding="utf8") as rf:
            for line in rf:
                ret.add(line.strip())
        return ret
    da = read_dic(Path(a))
    db = read_dic(Path(b))
    assert len(da) == len(db)
    assert len(da - db) == 0


if __name__ == '__main__':
    app()
