import json
from pathlib import Path
import time
from typing import Any, Dict, List, MutableMapping, Optional, Tuple
from rich.progress import Progress
import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import multiprocessing as mp

from loguru import logger
from alchemy.pipeline import EndStepPipeline, BeginStepPipeline, EvalPipeline, SchedPipeline
from alchemy.runner import get_dataloader
from alchemy.util.extention.rich import full_columns
from alchemy.util import warn_unused_kwargs
from alchemy import sym_tbl
import typer


@SchedPipeline.register()
class DumpOutputBSPipeline(BeginStepPipeline):
    def __init__(self, when: List[int], save_dir: str, split: str = "train", normalize: bool = True, **kwargs) -> None:
        super().__init__()
        # warn_unused_kwargs(kwargs)
        self.when = set(when)
        self.split = split
        self.save_dir = save_dir
        self.normalize = normalize

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
            sym_tbl().model.eval()
            with torch.no_grad():
                sample_ids = []
                hidden_lst = []
                for batch in itr:
                    eval_outputs = sym_tbl().model.forward(
                        batch,
                        needs_loss=False,
                        requires_grad=False,
                    )
                    self.__trans_outputs(batch, eval_outputs["hidden"].detach().cpu(), sample_ids, hidden_lst)
                    pbar.advance(tid)

            save_dir = record_dir / self.save_dir
            save_dir.mkdir(exist_ok=True)
            sample_save_path = save_dir / "sample{}.json".format(sym_tbl().train_sched.cur_step)
            hidden_save_path = sample_save_path.with_suffix(".npy")
            logger.info("Save samples ids at {}".format(sample_save_path))
            with sample_save_path.open('w', encoding="utf8") as wf:
                json.dump(sample_ids, wf, ensure_ascii=False)
            logger.info("Save hiddens at {}".format(hidden_save_path))
            np.save(hidden_save_path, hidden_lst, allow_pickle=True)
            if new_pbar:
                pbar.stop()
        return kwargs

    def __trans_outputs(self, batch: Dict[str, Any], hiddens: Tensor, sample_ids: List[str], hidden_lst: List[np.ndarray]):
        if self.normalize:
            hiddens = F.normalize(hiddens, dim=-1)
        for raw_sample, token_mask, hidden in zip(batch["raw_sample"], batch["token_masks"], hiddens):
            hidden = torch.masked_select(hidden, mask=token_mask.unsqueeze(-1)).view(-1, hidden.size(-1))
            sample_ids.append(raw_sample.id)
            hidden_lst.append(hidden.numpy())


@SchedPipeline.register()
class DumpProtoBSPipeline(BeginStepPipeline):
    def __init__(self, when: List[int], save_dir: str, normalize: bool = True, **kwargs) -> None:
        super().__init__()
        # warn_unused_kwargs(kwargs)
        self.when = set(when)
        self.save_dir = Path(save_dir)
        self.normalize = normalize

    def __call__(self, outputs: MutableMapping[str, Any], **kwargs) -> Dict[str, Any]:
        cur_step = sym_tbl().train_sched.cur_step
        record_dir: Optional[Path] = sym_tbl().try_get_global("record_dir")
        if cur_step in self.when and record_dir is not None:
            logger.info(
                "Dump proto at epoch {} step {}".format(sym_tbl().train_sched.cur_epoch, sym_tbl().train_sched.cur_step)
            )

            save_dir = record_dir / self.save_dir
            save_dir.mkdir(exist_ok=True)
            proto_save_path = save_dir / "proto{}.npy".format(sym_tbl().train_sched.cur_step)
            proto = sym_tbl().model.model.type_ebd.detach().cpu()
            if self.normalize:
                proto = F.normalize(proto, dim=-1)
            proto = proto.numpy()
            logger.info("Save proto ({}) at {}".format(proto.shape, proto_save_path))
            np.save(proto_save_path, proto)
        return kwargs


def tsne(paths: List[str], cuda: bool = True, metric: str = "innerproduct"):
    import re
    path_with_step = []
    for path in paths:
        path: Path = Path(path)
        match = re.fullmatch("sample(\d+)", path.stem)
        if match is None:
            continue
        path_with_step.append((path, match.group(1)))
    if cuda:
        tsne_cuda(path_with_step, metric)
    else:
        processes = []
        for path, step in path_with_step:
            ctx = mp.get_context('spawn')
            p = ctx.Process(target=tsne_sklearn_mp, args=(path, step, metric))
            processes.append(p)
            p.start()
            time.sleep(2)
        for p in processes:
            p.join()


def tsne_sklearn_mp(path: Path, step: str, metric: str = "innerproduct"):
    from sklearn.manifold import TSNE
    print("Loading from {}".format(path))
    hiddens = np.load(path, allow_pickle=True)
    hiddens = np.concatenate(hiddens, axis=0)
    print("hiddens shape = {}".format(hiddens.shape))

    proto_path = path.with_name("proto{}.npy".format(step))
    if proto_path.exists():
        print("Loading proto from {}".format(proto_path))
        proto = np.load(proto_path)
        print("proto shape = {}".format(proto.shape))
        hiddens = np.concatenate([hiddens, proto], axis=0)

    if metric == "innerproduct":
        metric = "cosine"
    elif metric == "l2":
        metric = "euclidean"
    else:
        raise NotImplementedError("Metric {} not implemented".format(metric))
    tsne = TSNE(n_components=2, init="pca", learning_rate="auto", metric=metric)
    tsne_hiddens = tsne.fit_transform(hiddens)
    if proto_path.exists():
        tsne_proto = tsne_hiddens[-proto.shape[0]:]
        print("TSNE proto shape = {}".format(tsne_proto.shape))
        dest = proto_path.with_name(proto_path.stem + "_tsne.npy")
        print("Save proto at {}".format(dest))
        np.save(dest, tsne_proto)

        tsne_hiddens = tsne_hiddens[:-proto.shape[0]]
    print("TSNE hiddens shape = {}".format(tsne_hiddens.shape))
    dest = path.with_name(path.stem + "_tsne.npy")
    print("Save hiddens at {}".format(dest))
    np.save(dest, tsne_hiddens)


def tsne_cuda(path_with_step: List[Tuple[Path, str]], metric: str = "innerproduct"):
    from tsnecuda import TSNE
    if metric == "innerproduct":
        metric = "innerproduct"
    elif metric == "l2":
        metric = "euclidean"
    else:
        raise NotImplementedError("Metric {} not implemented".format(metric))

    for path, step in path_with_step:
        print("Loading from {}".format(path))
        hiddens = np.load(path, allow_pickle=True)
        hiddens = np.concatenate(hiddens, axis=0)
        print("hiddens shape = {}".format(hiddens.shape))

        proto_path = path.with_name("proto{}.npy".format(step))
        if proto_path.exists():
            print("Loading proto from {}".format(proto_path))
            proto = np.load(proto_path)
            print("proto shape = {}".format(proto.shape))
            hiddens = np.concatenate([hiddens, proto], axis=0)

        tsne = TSNE(n_components=2, metric=metric)
        tsne_hiddens = tsne.fit_transform(hiddens)
        if proto_path.exists():
            tsne_proto = tsne_hiddens[-proto.shape[0]:]
            print("TSNE proto shape = {}".format(tsne_proto.shape))
            dest = proto_path.with_name(proto_path.stem + "_tsne.npy")
            print("Save proto at {}".format(dest))
            np.save(dest, tsne_proto)

            tsne_hiddens = tsne_hiddens[:-proto.shape[0]]
        print("TSNE hiddens shape = {}".format(tsne_hiddens.shape))
        dest = path.with_name(path.stem + "_tsne.npy")
        print("Save hiddens at {}".format(dest))
        np.save(dest, tsne_hiddens)


if __name__ == "__main__":
    typer.run(tsne)
