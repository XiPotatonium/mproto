"""
把原来的xx_types.json和xx_pos.json合并成xx_meta.json
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer

from .. import prepare_logging


app = typer.Typer()


@app.command()
def trans(
    datasets: List[str],
    types: str = typer.Option(...),
    output: str = typer.Option(...)
):
    output: Path = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    def mapper(sample: Dict[str, Any], id: str):
        new_sample = {
            "tokens": sample["tokens"],
            "entities": [
                {
                    "start": entity["start"],
                    "end": entity["end"],
                    # "type": "ENTITY",
                    "type": entity["type"]
                } for entity in sample["entities"]
            ],
        }
        if "ltokens" in sample:
            new_sample["ltokens"] = sample["ltokens"]
        if "rtokens" in sample:
            new_sample["rtokens"] = sample["rtokens"]
        new_sample["id"] = id
        return new_sample

    for datafile in datasets:
        datafile: Path = Path(datafile)
        if datafile.suffix == ".json":
            with datafile.open('r', encoding="utf8") as f:
                data_itr = json.load(f)
            with (output / (datafile.stem + ".jsonl")).open('w', encoding="utf8") as f:
                for i, sample in enumerate(data_itr):
                    sample = mapper(sample, "{}{}".format(datafile.stem, i))
                    f.write(json.dumps(sample, ensure_ascii=False))
                    f.write('\n')
        elif datafile.suffix == ".jsonl":
            with datafile.open('r', encoding="utf8") as f, \
                (output / datafile.name).open('w', encoding="utf8") as f_out:
                for i, line in enumerate(f):
                    sample = mapper(json.loads(line), "{}{}".format(datafile.stem, i))
                    f_out.write(json.dumps(sample, ensure_ascii=False))
                    f_out.write('\n')
        else:
            raise ValueError("Unknown data file format {}, expect .json or .jsonl".format(datafile))

    meta_info = {}
    with Path(types).open('r', encoding="utf8") as f:
        entities_types = json.load(f)["entities"]
    meta_info["entities"] = entities_types

    with (output / "meta.json").open('w', encoding="utf8") as f:
        json.dump(meta_info, f, ensure_ascii=False)


@app.command()
def json2jsonl(paths: List[str]):
    for path in paths:
        path: Path = Path(path)
        with path.open('r', encoding="utf8") as f:
            data = json.load(f)
        with path.with_suffix(".jsonl").open('w', encoding="utf8") as f:
            for i, instance in enumerate(data):
                f.write(json.dumps(instance, ensure_ascii=False) + '\n')


@app.command()
def filter_empty_gts(file: str, dest: Optional[str] = None):
    console = prepare_logging()
    file: Path = Path(file)
    samples = []
    with file.open('r', encoding="utf8") as rf:
        for line in rf:
            sample = json.loads(line)
            samples.append(sample)
    console.print("Get {} samples".format(len(samples)))
    samples = list(filter(lambda s: len(s["entities"]) != 0, samples))
    console.print("Get {} samples after filtering".format(len(samples)))

    if dest is None:
        dest: Path = file.with_stem(file.stem + "-wo-no-gts")
    else:
        dest = Path(dest)
    dest.parent.mkdir(exist_ok=True, parents=True)
    console.print("Save samples at {}".format(dest))

    with dest.open('w', encoding="utf8") as wf:
        for sample in samples:
            wf.write(json.dumps(sample, ensure_ascii=False))
            wf.write('\n')


if __name__ == '__main__':
    app()
