from collections import Counter
import json
from pathlib import Path
import typer

app = typer.Typer()


@app.command()
def main(src: str, target: str = typer.Option(...)):
    """将小类聚成大类，但是不要丢失小类的标注

    Args:
        src (str): _description_
        target (str, optional): _description_. Defaults to typer.Option(...).
    """
    def file_mapper(file: Path):
        with file.open('r', encoding="utf8") as rf:
            for line in rf:
                sample = json.loads(line)
                for e in sample["entities"]:
                    ty = e["type"]
                    ty1, ty2 = ty.split('-')
                    e["raw-type"] = e["type"]
                    e["type"] = ty1
                yield sample

    target_dir = Path(target)
    target_dir.mkdir(exist_ok=True, parents=True)
    for f in Path(src).iterdir():
        if f.suffix == ".jsonl":
            with (target_dir / f.name).open('w', encoding="utf8") as wf:
                for sample in file_mapper(f):
                    wf.write(json.dumps(sample, ensure_ascii=False))
                    wf.write("\n")

    with (Path(src) / "meta.json").open('r', encoding="utf8") as rf:
        meta = json.load(rf)
    meta["raw-entities"] = meta["entities"]
    meta["entities"] = {
        t: {"verbose": t, "short": t} for t in {k.split('-')[0] for k in meta["entities"].keys()}
    }
    with (target_dir / "meta.json").open('w', encoding="utf8") as wf:
        json.dump(meta, wf, ensure_ascii=False)
    coarse_counter = Counter(k.split('-')[0] for k in meta["raw-entities"].keys())
    print(coarse_counter)


if __name__ == "__main__":
    app()
