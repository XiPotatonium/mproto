import json
from pathlib import Path
import shutil
from typing import Any, Dict, List

from loguru import logger
import numpy as np

from . import prfs
from .. import prepare_logging
import typer
from rich.console import Console


app = typer.Typer()


def __ours2bond(sample: Dict[str, Any], tag2idx: Dict[str, int]):
    tokens = sample["tokens"]
    labels = np.zeros(len(tokens), dtype=int)
    for entity in sample["entities"]:
        labels[entity["start"]] = tag2idx["B-{}".format(entity["type"])]
        labels[entity["start"] + 1: entity["end"]] = tag2idx["I-{}".format(entity["type"])]

    return {
        "str_words": tokens,
        "words": [0] * len(tokens),
        "chars": [[0] * len(token) for token in tokens],
        "tags": [int(tag) for tag in labels],
        "defs": [None] * len(tokens)
    }


@app.command()
def ours2bond(dataset_dir: str, output_dir: str = typer.Option(...)):
    console = prepare_logging()
    dataset_dir = Path(dataset_dir)
    output_dir: Path = Path(output_dir)
    with (dataset_dir / "meta.json").open('r', encoding="utf8") as f:
        meta = json.load(f)
    tag2idx = {"O": 0}
    for ty in meta["entities"].keys():
        tag2idx["B-{}".format(ty)] = len(tag2idx)
        tag2idx["I-{}".format(ty)] = len(tag2idx)
    logger.info("tag2idx = {}".format(tag2idx))

    mapper = lambda x: __ours2bond(x, tag2idx)

    with (dataset_dir / "train.ds.json").open('r', encoding="utf8") as f:
        train_samples = json.load(f)
    with (dataset_dir / "dev.json").open('r', encoding="utf8") as f:
        dev_samples = json.load(f)
    with (dataset_dir / "test.json").open('r', encoding="utf8") as f:
        test_samples = json.load(f)
    train_samples = list(map(mapper, train_samples))
    dev_samples = list(map(mapper, dev_samples))
    test_samples = list(map(mapper, test_samples))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "train.json").open('w', encoding="utf8") as f:
        json.dump(train_samples, f)
    with (output_dir / "dev.json").open('w', encoding="utf8") as f:
        json.dump(dev_samples, f)
    with (output_dir / "test.json").open('w', encoding="utf8") as f:
        json.dump(test_samples, f)
    with (output_dir / "tag_to_id.json").open('w', encoding="utf8") as f:
        json.dump(tag2idx, f)


def __bond2ours(sample: Dict[str, Any], idx2tag: Dict[int, str]):
    tokens = sample["str_words"]
    tags = [idx2tag[tag] for tag in sample["tags"]]

    # 将gt_labels转换为gt_entities
    entities = []
    last_b_idx = -1
    for i, tag in enumerate(tags):
        if tag.startswith("I"):
            pass
        elif tag == "O":
            if last_b_idx >= 0:
                # 一个span结尾
                entities.append({"start": last_b_idx, "end": i, "type": tags[last_b_idx][2:]})
                last_b_idx = -1
        else:
            if last_b_idx >= 0:  # 一个span结尾
                entities.append({"start": last_b_idx, "end": i, "type": tags[last_b_idx][2:]})
                last_b_idx = -1
            last_b_idx = i
    if last_b_idx >= 0:
        entities.append({"start": last_b_idx, "end": len(tags), "type": tags[last_b_idx][2:]})
    return {"tokens": tokens, "entities": entities}


@app.command()
def bond2ours(dataset_dir: str, output_dir: str = typer.Option(...), standard_train_path: str = "data/datasets/conll03/dict_1.0/train.json"):
    console = prepare_logging()
    dataset_dir = Path(dataset_dir)
    output_dir: Path = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (dataset_dir / "tag_to_id.json").open('r', encoding="utf8") as f:
        tag2idx = json.load(f)
    logger.info("tag2idx = {}".format(tag2idx))
    idx2tag = {v: k for k, v in tag2idx.items()}
    entities = set()
    for tag in tag2idx.keys():
        if tag.startswith("B-"):
            entities.add(tag[2:])

    mapper = lambda x: __bond2ours(x, idx2tag)

    with (dataset_dir / "train.json").open('r', encoding="utf8") as f:
        train_samples = json.load(f)
    with (dataset_dir / "dev.json").open('r', encoding="utf8") as f:
        dev_samples = json.load(f)
    with (dataset_dir / "test.json").open('r', encoding="utf8") as f:
        test_samples = json.load(f)

    train_samples = list(map(mapper, train_samples))
    dev_samples = list(map(mapper, dev_samples))
    test_samples = list(map(mapper, test_samples))

    with (output_dir / "train.jsonl").open('w', encoding="utf8") as f:
        logger.info("Got {} train ds samples".format(len(train_samples)))
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False))
            f.write('\n')
    with (output_dir / "dev.jsonl").open('w', encoding="utf8") as f:
        logger.info("Got {} dev samples".format(len(dev_samples)))
        for sample in dev_samples:
            f.write(json.dumps(sample, ensure_ascii=False))
            f.write('\n')
    with (output_dir / "test.jsonl").open('w', encoding="utf8") as f:
        logger.info("Got {} test samples".format(len(test_samples)))
        for sample in test_samples:
            f.write(json.dumps(sample, ensure_ascii=False))
            f.write('\n')
    with (output_dir / "meta.json").open('w', encoding="utf8") as f:
        json.dump({"entities": {ty: {"short": ty, "verbose": ty} for ty in entities}}, f)

    with Path(standard_train_path).open('r', encoding="utf8") as f:
        std_train = json.load(f)
        if len(std_train) != len(train_samples):
            raise ValueError("Mismatch")
        for s0, s1 in zip(std_train, train_samples):
            if len(s0["tokens"]) != s1["tokens"]:
                raise ValueError("Mismathch")
            if not all(t0 == t1 for t0, t1 in zip(s0["tokens"], s1["tokens"])):
                raise ValueError("Mismatch")
        label_mapper = lambda es: [(e["start"], e["end"], e["type"]) for e in es]
        prfs(
            [label_mapper(sample["entities"]) for sample in std_train],
            [label_mapper(sample["entities"]) for sample in train_samples],
            console=console
        )


if __name__ =="__main__":
    app()
