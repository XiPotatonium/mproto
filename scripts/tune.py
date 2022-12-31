import itertools
import json
from pathlib import Path
import re
from typing import List, Optional, Union
from rich.console import Console
from rich.table import Table
import tomlkit

import typer
from alchemy import prepare_cfg, run
from alchemy.runner import RunResult
from .train_and_test import train_and_test


app = typer.Typer()


def display(results: List[Union[RunResult, Exception]]):
    records = [result.record_dir for result in results if isinstance(result, RunResult)]
    records.sort()
    console = Console()
    table = Table()
    table.add_column("tag")
    table.add_column("step")
    table.add_column("value")
    for record_dir in records:
        with (record_dir / "best_info.json").open('r', encoding="utf8") as rf:
            best_info = json.load(rf)
        with (record_dir / "cfg.toml").open('r', encoding="utf8") as rf:
            record_cfg = tomlkit.load(rf)
        table.add_row(record_cfg["tag"], str(best_info["step"]), str(round(best_info["f1_micro"], 4)))
    console.print(table)


@app.command()
def tune_none(
    cfg_paths: List[str],
    device: Optional[List[int]] = None,
    user_dir: str = "src",
    desc: str = "",
    debug: bool = False,
    task_per_device: int = 1,
    test: bool = True,
):
    none_ratios = [0.008, 0.01, 0.02]

    cfgs = []
    for (i, cfg_path), none_ratio in itertools.product(
        enumerate(cfg_paths), none_ratios
    ):
        cfg_toml = prepare_cfg(Path(cfg_path))
        for plugin in cfg_toml["plugins"]:
            if plugin["type"] == "alchemy.plugins.FileLogger":
                plugin["log_dir"] = str(Path(plugin["log_dir"]).with_name("try"))
        if not test:
            evalpipes = cfg_toml["task"]["evalpipes"]
            for pipe in list(evalpipes):       # 浅拷贝一份，防止remove in iter的问题
                if pipe["type"] == "src.task.ner.evalpipe.SaveModel":
                    # 调参就不要保存模型了，省点空间
                    evalpipes.remove(pipe)
        cfg_toml["tag"] += "-try{}-n{:.3f}".format(i, none_ratio)
        cfg_toml["criterion"]["none_ratio"] = none_ratio
        cfgs.append(cfg_toml)

    if test:
        results = train_and_test(
            cfgs=cfgs,
            device=device,
            user_dir=user_dir,
            desc=desc,
            debug=debug,
            task_per_device=task_per_device,
        )
    else:
        results = run(
            cfgs=cfgs,
            device=device,
            user_dir=user_dir,
            desc=desc,
            debug=debug,
            task_per_device=task_per_device,
        )

    display(results)


@app.command()
def tune_proto(
    cfg_paths: List[str],
    device: Optional[List[int]] = None,
    user_dir: str = "src",
    desc: str = "",
    debug: bool = False,
    task_per_device: int = 1,
    test: bool = True,
):
    protos = [1, 2, 3, 4, 5, 6]

    cfgs = []
    for (i, cfg_path), p in itertools.product(
        enumerate(cfg_paths), protos
    ):
        cfg_toml = prepare_cfg(Path(cfg_path))
        for plugin in cfg_toml["plugins"]:
            if plugin["type"] == "alchemy.plugins.FileLogger":
                plugin["log_dir"] = str(Path(plugin["log_dir"]).with_name("try"))
        if not test:
            evalpipes = cfg_toml["task"]["evalpipes"]
            for pipe in list(evalpipes):       # 浅拷贝一份，防止remove in iter的问题
                if pipe["type"] == "src.task.ner.evalpipe.SaveModel":
                    # 调参就不要保存模型了，省点空间
                    evalpipes.remove(pipe)
        cfg_toml["tag"] += "-try{}-p{}".format(i, p)
        cfg_toml["model"]["num_proto_per_type"] = p
        cfgs.append(cfg_toml)
    if test:
        results = train_and_test(
            cfgs=cfgs,
            device=device,
            user_dir=user_dir,
            desc=desc,
            debug=debug,
            task_per_device=task_per_device,
        )
    else:
        results = run(
            cfgs=cfgs,
            device=device,
            user_dir=user_dir,
            desc=desc,
            debug=debug,
            task_per_device=task_per_device,
        )

    display(results)


@app.command()
def tune_proto_none(
    cfg_paths: List[str],
    device: Optional[List[int]] = None,
    user_dir: str = "src",
    desc: str = "",
    debug: bool = False,
    task_per_device: int = 1,
    test: bool = True,
):
    protos = [1, 2, 3, 4, 5]
    none_ratios = [0.008, 0.01, 0.02, 0.03, 0.05]

    cfgs = []
    for (i, cfg_path), p, none_ratio in itertools.product(
        enumerate(cfg_paths), protos, none_ratios
    ):
        cfg_toml = prepare_cfg(Path(cfg_path))
        for plugin in cfg_toml["plugins"]:
            if plugin["type"] == "alchemy.plugins.FileLogger":
                plugin["log_dir"] = str(Path(plugin["log_dir"]).with_name("try"))
        if not test:
            evalpipes = cfg_toml["task"]["evalpipes"]
            for pipe in list(evalpipes):       # 浅拷贝一份，防止remove in iter的问题
                if pipe["type"] == "src.task.ner.evalpipe.SaveModel":
                    # 调参就不要保存模型了，省点空间
                    evalpipes.remove(pipe)
        cfg_toml["tag"] += "-try{}-p{}n{:.3f}".format(i, p, none_ratio)
        cfg_toml["model"]["num_proto_per_type"] = p
        cfg_toml["criterion"]["none_ratio"] = none_ratio
        cfgs.append(cfg_toml)
    if test:
        results = train_and_test(
            cfgs=cfgs,
            device=device,
            user_dir=user_dir,
            desc=desc,
            debug=debug,
            task_per_device=task_per_device,
        )
    else:
        results = run(
            cfgs=cfgs,
            device=device,
            user_dir=user_dir,
            desc=desc,
            debug=debug,
            task_per_device=task_per_device,
        )

    display(results)


@app.command()
def tune_all(
    cfg_paths: List[str],
    device: Optional[List[int]] = None,
    user_dir: str = "src",
    desc: str = "",
    debug: bool = False,
    task_per_device: int = 1,
    test: bool = True,
):
    protos = [3, 4]
    none_ratios = [0.008, 0.01, 0.02, 0.03, 0.05]
    compact_weights = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
    momentums = [0.5, 0.7, 0.9]

    cfgs = []
    for (i, cfg_path), p, none_ratio, compact_weight, momentum in itertools.product(
        enumerate(cfg_paths), protos, none_ratios, compact_weights, momentums
    ):
        cfg_toml = prepare_cfg(Path(cfg_path))
        for plugin in cfg_toml["plugins"]:
            if plugin["type"] == "alchemy.plugins.FileLogger":
                plugin["log_dir"] = str(Path(plugin["log_dir"]).with_name("try"))
        if not test:
            evalpipes = cfg_toml["task"]["evalpipes"]
            for pipe in list(evalpipes):       # 浅拷贝一份，防止remove in iter的问题
                if pipe["type"] == "src.task.ner.evalpipe.SaveModel":
                    # 调参就不要保存模型了，省点空间
                    evalpipes.remove(pipe)
        cfg_toml["tag"] += "-try{}-p{}n{:.3f}c{:.3f}m{:.1f}".format(i, p, none_ratio, compact_weight, momentum)
        cfg_toml["model"]["num_proto_per_type"] = p
        cfg_toml["criterion"]["none_ratio"] = none_ratio
        cfg_toml["criterion"]["compact_weight"] = compact_weight
        cfg_toml["criterion"]["proto_ema_momentum"] = momentum
        cfgs.append(cfg_toml)
    if test:
        results = train_and_test(
            cfgs=cfgs,
            device=device,
            user_dir=user_dir,
            desc=desc,
            debug=debug,
            task_per_device=task_per_device,
        )
    else:
        results = run(
            cfgs=cfgs,
            device=device,
            user_dir=user_dir,
            desc=desc,
            debug=debug,
            task_per_device=task_per_device,
        )

    display(results)


@app.command()
def tune_criterion(
    cfg_paths: List[str],
    device: Optional[List[int]] = None,
    user_dir: str = "src",
    desc: str = "",
    debug: bool = False,
    task_per_device: int = 1,
    test: bool = True,
):
    none_ratios = [0.008, 0.01, 0.02, 0.03, 0.05]
    compact_weights = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
    momentums = [0.5, 0.7, 0.9]

    cfgs = []
    for (i, cfg_path), none_ratio, compact_weight, momentum in itertools.product(
        enumerate(cfg_paths), none_ratios, compact_weights, momentums
    ):
        cfg_toml = prepare_cfg(Path(cfg_path))
        for plugin in cfg_toml["plugins"]:
            if plugin["type"] == "alchemy.plugins.FileLogger":
                plugin["log_dir"] = str(Path(plugin["log_dir"]).with_name("try"))
        if not test:
            evalpipes = cfg_toml["task"]["evalpipes"]
            for pipe in list(evalpipes):       # 浅拷贝一份，防止remove in iter的问题
                if pipe["type"] == "src.task.ner.evalpipe.SaveModel":
                    # 调参就不要保存模型了，省点空间
                    evalpipes.remove(pipe)
        cfg_toml["tag"] += "-try{}-n{:.3f}c{:.3f}m{:.1f}".format(i, none_ratio, compact_weight, momentum)
        cfg_toml["criterion"]["none_ratio"] = none_ratio
        cfg_toml["criterion"]["compact_weight"] = compact_weight
        cfg_toml["criterion"]["proto_ema_momentum"] = momentum
        cfgs.append(cfg_toml)
    if test:
        results = train_and_test(
            cfgs=cfgs,
            device=device,
            user_dir=user_dir,
            desc=desc,
            debug=debug,
            task_per_device=task_per_device,
        )
    else:
        results = run(
            cfgs=cfgs,
            device=device,
            user_dir=user_dir,
            desc=desc,
            debug=debug,
            task_per_device=task_per_device,
        )

    display(results)


@app.command()
def tune_seed(
    cfg_paths: List[str],
    device: Optional[List[int]] = None,
    user_dir: str = "src",
    desc: str = "",
    debug: bool = False,
    task_per_device: int = 1,
    test: bool = True,
):
    seeds = [0, 1, 2, 3, 4]

    cfgs = []
    for (i, cfg_path), seed in itertools.product(
        enumerate(cfg_paths), seeds
    ):
        cfg_toml = prepare_cfg(Path(cfg_path))
        has_seeding_plugin = False
        for plugin in cfg_toml["plugins"]:
            if plugin["type"] == "alchemy.plugins.FileLogger":
                plugin["log_dir"] = str(Path(plugin["log_dir"]).with_name("try"))
            if plugin["type"] == "alchemy.plugins.Seeding":
                plugin["seed"] = seed
                has_seeding_plugin = True
        if not has_seeding_plugin:
            raise RuntimeError("No seeding plugin")
        if not test:
            evalpipes = cfg_toml["task"]["evalpipes"]
            for pipe in list(evalpipes):       # 浅拷贝一份，防止remove in iter的问题
                if pipe["type"] == "src.task.ner.evalpipe.SaveModel":
                    # 调参就不要保存模型了，省点空间
                    evalpipes.remove(pipe)
        cfg_toml["tag"] += "-try{}-s{}".format(i, seed)
        cfgs.append(cfg_toml)
    if test:
        results = train_and_test(
            cfgs=cfgs,
            device=device,
            user_dir=user_dir,
            desc=desc,
            debug=debug,
            task_per_device=task_per_device,
        )
    else:
        results = run(
            cfgs=cfgs,
            device=device,
            user_dir=user_dir,
            desc=desc,
            debug=debug,
            task_per_device=task_per_device,
        )

    display(results)


@app.command()
def ab(
    cfg_paths: List[str],
    device: Optional[List[int]] = None,
    user_dir: str = "src",
    desc: str = "",
    debug: bool = False,
    task_per_device: int = 1,
    test: bool = True,
):
    cfgs = []
    for (i, cfg_path), with_mproto, with_denoised_ot, with_compact_loss in itertools.product(
        enumerate(cfg_paths), [True, False], [True, False], [True, False]
    ):
        cfg_toml = prepare_cfg(Path(cfg_path))
        for plugin in cfg_toml["plugins"]:
            if plugin["type"] == "alchemy.plugins.FileLogger":
                plugin["log_dir"] = str(Path(plugin["log_dir"]).with_name("ab"))
        if not test:
            evalpipes = cfg_toml["task"]["evalpipes"]
            for pipe in list(evalpipes):       # 浅拷贝一份，防止remove in iter的问题
                if pipe["type"] == "src.task.ner.evalpipe.SaveModel":
                    # 调参就不要保存模型了，省点空间
                    evalpipes.remove(pipe)
        if not with_mproto:
            cfg_toml["model"]["num_proto_per_type"] = 1
        if not with_denoised_ot:
            cfg_toml["model"]["type"] = "src.models.mproto.BaseMProtoTagger"
        if not with_compact_loss:
            cfg_toml["criterion"]["compact_weight"] = 0.0
        cfg_toml["tag"] += "-ab{}-{}mp-{}dot-{}c".format(
            i,
            "w" if with_mproto else "wo",
            "w" if with_denoised_ot else "wo",
            "w" if with_compact_loss else "wo",
        )
        cfgs.append(cfg_toml)
    if test:
        results = train_and_test(
            cfgs=cfgs,
            device=device,
            user_dir=user_dir,
            desc=desc,
            debug=debug,
            task_per_device=task_per_device,
        )
    else:
        results = run(
            cfgs=cfgs,
            device=device,
            user_dir=user_dir,
            desc=desc,
            debug=debug,
            task_per_device=task_per_device,
        )

    display(results)


@app.command()
def repro(
    cfg: str,
    device: Optional[List[int]] = None,
    user_dir: str = "src",
    desc: str = "",
    debug: bool = False,
    file: bool = False,
):
    def read_cfg_for_repro(tag: str):
        cfg_toml = prepare_cfg(Path(cfg))
        for plugin in cfg_toml["plugins"]:
            if plugin["type"] == "alchemy.plugins.FileLogger":
                plugin["log_dir"] = str(Path(plugin["log_dir"]).with_name("repro"))
        cfg_toml["tag"] += "-{}".format(tag)
        sched_pipes = cfg_toml["sched"]["pipes"]
        for pipe in sched_pipes:
            if pipe["type"] == "src.pipeline.LogLRESPipeline":
                pipe["log_file"] = True
                break
        else:
            sched_pipes.append({"type": "src.pipeline.LogLRESPipeline", "log_tensorboard": False, "log_file": True})
        return cfg_toml
    cfgs = [read_cfg_for_repro("repro1"), read_cfg_for_repro("repro2")]
    run(
        cfgs=cfgs,
        device=device,
        user_dir=user_dir,
        desc=desc,
        debug=debug,
        no_file=not file,
    )


@app.command()
def show_tune_result(records: List[str], metric: str = "f1_micro", find: Optional[str] = None):
    console = Console()
    table = Table()
    table.add_column("tag")
    table.add_column("step")
    table.add_column("value")
    table.add_column("dir")
    records = list(records)     # 好像其实typer给的是tuple
    records.sort()
    for record_dir in records:
        record_dir: Path = Path(record_dir)
        best_info_path = record_dir / "best_info.json"
        if not best_info_path.exists():
            # 有一些情况可能会没有best_info，例如刚刚开始训练
            console.print("WARNING: missing {}. Skipped".format(best_info_path))
            continue
        with best_info_path.open('r', encoding="utf8") as rf:
            best_info = json.load(rf)
        with (record_dir / "cfg.toml").open('r', encoding="utf8") as rf:
            record_cfg = tomlkit.load(rf)
        tag = record_cfg["tag"]
        if find is not None and not re.fullmatch(find, tag):
            console.print("Skip {} ({})".format(tag, record_dir))
            continue
        table.add_row(tag, str(best_info["step"]), str(round(best_info[metric], 4)), record_dir.name)
    console.print(table)


if __name__ == "__main__":
    app()
