import json
from pathlib import Path
import shutil
import typer


app = typer.Typer()


@app.command()
def trans(src: str, dest: str = typer.Option(...)):
    types = {"entities": {}}
    dest: Path = Path(dest)
    dest.mkdir(exist_ok=True, parents=True)

    doc_id = 0
    sample_id = 0
    for split in ["train", "dev", "test"]:
        split_sample_count = 0
        with (Path(src) / f"{split}.txt").open('r', encoding="utf8") as rf, \
                (dest / f"{split}.jsonl").open('w', encoding="utf8") as wf:
            sample = {"tokens": [], "entities": [], "pos": []}
            idx = 0
            start = end = None
            entity_type = None

            for line in rf:
                line = line.strip()
                if line == "-DOCSTART- -X- -X- O":
                    doc_id += 1
                    continue
                if line:
                    last = idx

                    fields = list(filter(lambda x: x, line.split(" ")))
                    # print(fields)
                    sample["tokens"].append(fields[0])
                    sample["pos"].append(fields[1])
                    sample["doc_id"] = doc_id
                    sample["id"] = sample_id
                    if fields[3].startswith("B-"):
                        if start != None and end != None and end == idx:
                            sample["entities"].append({"start": start, "end": end, "type": entity_type})
                        start = idx
                        end = idx + 1
                        entity_type = fields[3][2:]
                        if entity_type not in types["entities"]:
                            types["entities"][entity_type] = {"verbose": entity_type, "short": entity_type}
                    if fields[3].startswith("I-"):
                        end = end + 1
                    if fields[3] == "O" and start != None:
                        sample["entities"].append({"start": start, "end": end, "type": entity_type})
                        start = end = entity_type = None
                    idx += 1
                else:
                    if start != None:
                        sample["entities"].append({"start": start, "end": end, "type": entity_type})
                    idx = 0
                    start = end = None
                    entity_type = None
                    # if len(sample["tokens"]) and (len(sample["entities"])!=0 or ty!="train"):
                    if len(sample["tokens"]):
                        wf.write(json.dumps(sample))
                        wf.write('\n')
                        split_sample_count += 1
                        sample_id += 1

                    sample = {"tokens": [], "entities": [], "pos": []}
            if len(sample["tokens"]):
                wf.write(json.dumps(sample, ensure_ascii=False))
                wf.write('\n')
                split_sample_count += 1
                sample_id += 1

            print(split_sample_count)

    print(types)
    with (dest / "meta.json").open('w', encoding="utf8") as f:
        json.dump(types, f, ensure_ascii=False)


@app.command()
def trans_with_context(dset: str, dest: str = typer.Option(...), window: int = 2):
    dest: Path = Path(dest)
    dest.mkdir(exist_ok=True, parents=True)

    for file in Path(dset).iterdir():
        if file.suffix != ".jsonl":
            continue
        with file.open('r', encoding="utf8") as f:
            datasets = [json.loads(line) for line in f]
        with (dest / file.name).open('w', encoding="utf8") as f:
            for i, sample in enumerate(datasets):
                ltokens = []
                rtokens = []
                k = window - 1
                while k >= 0:
                    # doc_id需要一致，也就是说上下文需要是同一个文章里面的，不能跨越文章
                    if i > k and sample["doc_id"] == datasets[i - k - 1]["doc_id"]:
                        ltokens.extend(datasets[i - k - 1]["tokens"])
                    k -= 1
                k = 1
                while k < window + 1:
                    if i < len(datasets) - k and sample["doc_id"] == datasets[i + k]["doc_id"]:
                        rtokens.extend(datasets[i + k]["tokens"])
                    k += 1
                sample["ltokens"] = ltokens
                sample["rtokens"] = rtokens
                f.write(json.dumps(sample, ensure_ascii=False))
                f.write('\n')
    shutil.copy(str(Path(dset) / "meta.json"), str(Path(dest) / "meta.json"))


if __name__ == "__main__":
    app()
