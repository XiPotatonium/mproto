import json
from pathlib import Path
from typing import Any, Dict
from rich.progress import track
import typer
from loguru import logger

from .. import prepare_logging

app = typer.Typer()


def copy_anno(file: Path):
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

        sample = {
            "tokens": tokens,
            "fully_labels": fully_labels,
            "fully_entities": fully_entities,
        }
        samples.append(sample)

    return samples


@app.command()
def main(src: str, dest: str = typer.Option(...)):
    output_dir = Path(dest)
    dataset_dir = Path(src)
    console = prepare_logging()

    output_dir.mkdir(exist_ok=True, parents=True)

    trainFile = dataset_dir / 'train.txt'
    logger.info("Load train samples from \"{}\"".format(trainFile))
    train_samples = copy_anno(trainFile)
    # train_samples = tagging2span(train_samples, tag2idx)

    # 训练数据选择远程监督的结果作为gt
    with (output_dir / "train.jsonl").open('w', encoding="utf8") as wf:
        for i, sample in enumerate(train_samples):
            train_sample = {
                "tokens": sample["tokens"],
                "entities": sample["fully_entities"],
                "id": "train{}".format(i)
            }
            wf.write(json.dumps(train_sample, ensure_ascii=False))
            wf.write('\n')

    val_file = dataset_dir / 'valid.txt'
    logger.info("Load val samples from \"{}\"".format(val_file))
    val_samples = copy_anno(val_file)

    test_file = dataset_dir / 'test.txt'
    logger.info("Load test samples from \"{}\"".format(test_file))
    test_samples = copy_anno(test_file)

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

    # entities = ["Chemical", "Disease"]
    entities = ["PER", "LOC", "ORG", "MISC"]
    meta = {
        "entities": {ty: {"short": ty, "verbose": ty} for ty in entities},
    }
    with (output_dir / "meta.json").open('w', encoding="utf8") as wf:
        json.dump(meta, wf, ensure_ascii=False)


if __name__ == "__main__":
    app()
