import json
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, Optional, Tuple
from rich.progress import Progress
import torch
from torch import Tensor
import torch.nn.functional as F

import typer
from loguru import logger
from alchemy.pipeline import EndStepPipeline, BeginStepPipeline, EvalPipeline, SchedPipeline
from alchemy.runner import get_dataloader
from alchemy.util.extention.rich import full_columns
from alchemy import sym_tbl
import tomlkit
import numpy as np


app = typer.Typer()


@SchedPipeline.register()
class DumpBSPipeline(BeginStepPipeline):
    def __init__(
        self,
        when: List[int],
        save_dir: str,
        split: str = "train",
        **kwargs
    ) -> None:
        super().__init__()
        # warn_unused_kwargs(kwargs)
        self.when = set(when)
        self.split = split
        self.save_dir = save_dir
        self.assingment_var = "assignments"
        self.similarity_var = "similarity"

    def __call__(self, outputs: MutableMapping[str, Any], **kwargs) -> Dict[str, Any]:
        cur_step = sym_tbl().train_sched.cur_step
        record_dir: Optional[Path] = sym_tbl().try_get_global("record_dir")
        if cur_step in self.when and record_dir is not None:
            logger.info(
                "Dump output hidden at epoch {} step {}".format(sym_tbl().train_sched.cur_epoch, sym_tbl().train_sched.cur_step)
            )

            dset, dset_kw = sym_tbl().task.dataset(self.split)
            itr = get_dataloader(
                dset,
                **dset_kw,
            )

            pbar: Progress = sym_tbl().try_get_global("pbar")
            new_pbar = pbar is None
            if new_pbar:
                pbar = Progress(*full_columns(), console=sym_tbl().console, disable=sym_tbl().console is None)
                pbar.start()
            tid = pbar.add_task("Eval", total=len(itr))

            if sym_tbl().contains_global(self.assingment_var):
                logger.warning("Override existing {}".format(self.assingment_var))
            sym_tbl().set_global(self.assingment_var, [])
            if sym_tbl().contains_global(self.similarity_var):
                logger.warning("Override existing {}".format(self.similarity_var))
            sym_tbl().set_global(self.similarity_var, [])

            sym_tbl().model.eval()
            with torch.no_grad():
                sample_ids = []
                for batch in itr:
                    batch_sample_ids = []
                    _ = sym_tbl().model.forward(
                        batch,
                        needs_loss=True,
                        requires_grad=False,
                    )
                    for raw_sample in batch["raw_sample"]:
                        batch_sample_ids.append(raw_sample.id)
                    sample_ids.append(batch_sample_ids)
                    pbar.advance(tid)

            assignments: List[Tensor] = sym_tbl().pop_global(self.assingment_var)
            assignments: Tensor = torch.cat(assignments)        # N * P

            sims: List[Tensor] = sym_tbl().pop_global(self.similarity_var)
            sims: Tensor = torch.cat(sims)        # N * P

            save_dir = record_dir / self.save_dir
            save_dir.mkdir(exist_ok=True)
            sample_save_path = save_dir / "id{}.json".format(sym_tbl().train_sched.cur_step)
            assignments_save_path = save_dir / "assignments{}.npy".format(sym_tbl().train_sched.cur_step)
            sims_save_path = save_dir / "similarity{}.npy".format(sym_tbl().train_sched.cur_step)

            logger.info("Save samples ids at {}".format(sample_save_path))
            with sample_save_path.open('w', encoding="utf8") as wf:
                json.dump(sample_ids, wf, ensure_ascii=False)

            logger.info("Save assignments at {}".format(assignments_save_path))
            np.save(assignments_save_path, assignments.numpy(), allow_pickle=True)

            logger.info("Save similarity at {}".format(sims_save_path))
            np.save(sims_save_path, sims.numpy(), allow_pickle=True)

            if new_pbar:
                pbar.stop()
        return kwargs


@app.command()
def assignments(log_dir: str, step: int):

    ids_path = log_dir / "detail_log" / "id{}.json".format(step)
    assignments_path = ids_path.with_name("assignments{}_tsne.npy".format(step))

    with (log_dir / "cfg.toml").open('r', encoding="utf8") as rf:
        cfg = tomlkit.load(rf)
    num_proto_per_type = cfg["model"]["num_proto_per_type"]
    for pipe in cfg["task"]["datasets"]["train"]["pipes"]:
        if pipe["type"] == "alchemy.pipeline.itr.Batch":
            bsz = pipe["batch_size"]
        elif pipe["type"] == "alchemy.pipeline.lst.SequenceWrapper":
            samples_path = Path(pipe["datapipe"][0])
    meta_path = Path(cfg["task"]["meta"])

    with meta_path.open('r', encoding="utf8") as rf:
        meta = json.load(rf)
    id2type = ["None"]
    # specified entity types
    for i, (key, v) in enumerate(meta['entities'].items()):
        if key == "None":
            # 一般而言不会重载None
            pass
        else:
            id2type.append(key)
    type2id = {v: i for i, v in enumerate(id2type)}

    id2sample = {}
    with samples_path.open('r', encoding="utf8") as rf:
        for line in rf:
            sample = json.loads(line)
            labels = np.zeros(len(sample["tokens"]), dtype=int)
            for m in sample["entities"]:
                labels[m["start"]:m["end"]] = type2id[m["type"]]
            sample["labels"] = labels
            hu_labels = np.zeros(len(sample["tokens"]), dtype=int)
            for m in sample["human_entities"]:
                hu_labels[m["start"]:m["end"]] = type2id[m["type"]]
            sample["hu_labels"] = hu_labels
            id2sample[sample["id"]] = sample

    assignments = np.load(assignments_path)
    print(assignments.shape)
    batches = []
    batch_nones = []

    with ids_path.open('r', encoding="utf8") as rf:
        ids = json.load(rf)
        for batch_ids in ids:
            batches.append([])
            batch_nones.append(0)
            for sample_id in batch_ids:
                sample = id2sample[sample_id]
                batches[-1].append(sample)
                batch_nones[-1] += sum(sample["labels"] == 0)

    assert sum(batch_nones) == len(assignments)
    pivots = np.cumsum(np.asarray(batch_nones))
    assignments = np.split(assignments, pivots)
    confusion_mat = np.zeros((len(id2type) - 1, len(id2type)), dtype=float)
    for batch, b_assignments in zip(batches, assignments):
        hu_labels = []
        for sample in batch:
            for label, hu_label in zip(sample["labels"], sample["hu_labels"]):
                if label == 0:
                    hu_labels.append(hu_label)
        assert len(hu_labels) == b_assignments.shape[0], "{} != {}".format(
            b_assignments.shape, hu_labels
        )
        for assignment, hu_label in zip(b_assignments, hu_labels):
            assigned_pid = np.argmax(assignment)
            assigned_tid = assigned_pid // num_proto_per_type
            confusion_mat[hu_label, assigned_tid] += 1

    print(confusion_mat)


@app.command()
def similarity(log_dir: str):
    import re
    import pandas as pd

    with (Path(log_dir) / "cfg.toml").open('r', encoding="utf8") as rf:
        cfg = tomlkit.load(rf)
    num_proto_per_type = cfg["model"]["num_proto_per_type"]
    for pipe in cfg["task"]["datasets"]["train"]["pipes"]:
        if pipe["type"] == "alchemy.pipeline.itr.Batch":
            bsz = pipe["batch_size"]
        elif pipe["type"] == "alchemy.pipeline.lst.SequenceWrapper":
            samples_path = Path(pipe["datapipe"][0])
    meta_path = Path(cfg["task"]["meta"])

    with meta_path.open('r', encoding="utf8") as rf:
        meta = json.load(rf)
    id2type = ["None"]
    # specified entity types
    for i, (key, v) in enumerate(meta['entities'].items()):
        if key == "None":
            # 一般而言不会重载None
            pass
        else:
            id2type.append(key)
    type2id = {v: i for i, v in enumerate(id2type)}

    id2sample = {}
    with samples_path.open('r', encoding="utf8") as rf:
        for line in rf:
            sample = json.loads(line)
            labels = np.zeros(len(sample["tokens"]), dtype=int)
            for m in sample["entities"]:
                labels[m["start"]:m["end"]] = type2id[m["type"]]
            sample["labels"] = labels
            hu_labels = np.zeros(len(sample["tokens"]), dtype=int)
            for m in sample["human_entities"]:
                hu_labels[m["start"]:m["end"]] = type2id[m["type"]]
            sample["hu_labels"] = hu_labels
            id2sample[sample["id"]] = sample

    def itr_ids():
        for file in (Path(log_dir) / "detail_log").iterdir():
            m = re.fullmatch(r"id(\d+)\.json", file.name)
            if m is not None:
                yield file, m.group(1)

    data = {}

    for id_file, step in itr_ids():
        step = int(step)
        sims_path = id_file.with_name("similarity{}.npy".format(step))

        sims = np.load(sims_path)
        # print(f"min = {np.min(sims)}, max = {np.max(sims)}")
        # print(assignments.shape)
        batches = []
        batch_nones = []

        with id_file.open('r', encoding="utf8") as rf:
            ids = json.load(rf)
            for batch_ids in ids:
                batches.append([])
                batch_nones.append(0)
                for sample_id in batch_ids:
                    sample = id2sample[sample_id]
                    batches[-1].append(sample)
                    batch_nones[-1] += sum(sample["labels"] == 0)

        assert sum(batch_nones) == len(sims)
        pivots = np.cumsum(np.asarray(batch_nones))
        sims = np.split(sims, pivots)
        ty_count = np.zeros(len(id2type) + 2, dtype=int)        # +2 for macro and micro
        ty_sims = np.zeros_like(ty_count, dtype=float)
        for batch, b_sims in zip(batches, sims):
            hu_labels = []
            for sample in batch:
                for label, hu_label in zip(sample["labels"], sample["hu_labels"]):
                    if label == 0:
                        hu_labels.append(hu_label)
            assert len(hu_labels) == b_sims.shape[0], "{} != {}".format(
                b_sims.shape, hu_labels
            )
            for t_sim, hu_label in zip(b_sims, hu_labels):
                ty_sims[hu_label] += t_sim[hu_label]
                ty_sims[0] += t_sim[0]
                ty_count[hu_label] += 1
                ty_count[0] += 1

        ty_sims[-1] = np.sum(ty_sims[1:-2])
        ty_count[-1] = np.sum(ty_count[1:-2])
        ty_sims = ty_sims / ty_count
        ty_sims[-2] = np.mean(ty_sims[1:-2])
        data[step] = ty_sims
        # print(confusion_mat)
    print(ty_count)

    data = list(data.items())
    data.sort(key=lambda x: x[0])
    data = [[step, *sim] for step, sim in data]
    dframe = pd.DataFrame(data, columns=["step", *id2type, "macro", "micro"])
    print(dframe)
    dframe.to_csv(Path(log_dir) / "detail_log" / "similarity.csv", index=False)


if __name__ == "__main__":
    app()
