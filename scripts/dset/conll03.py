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


if __name__ == "__main__":
    app()
