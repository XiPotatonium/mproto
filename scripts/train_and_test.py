from pathlib import Path
from typing import List, Optional
import tomlkit

import typer
from alchemy import prepare_cfg, run


app = typer.Typer()


def train_and_test(
    cfgs: List[tomlkit.TOMLDocument],
    device: Optional[List[int]] = None,
    user_dir: str = "src",
    desc: str = "",
    debug: bool = False,
    task_per_device: int = 1,
):
    run_results = run(
        cfgs=cfgs,
        device=device,
        user_dir=user_dir,
        desc=desc,
        debug=debug,
        task_per_device=task_per_device,
    )

    test_cfgs = []
    # 因为return返回的顺序可能和cfg不同，所以这里再次从record_dir中读取cfg
    for run_result in run_results:
        if isinstance(run_result, Exception):
            # 遇到exception
            raise run_result
        cfg = prepare_cfg(run_result.record_dir / "cfg.toml")

        cfg["runner"] = "alchemy.runner.Tester"     # 将runner指定为Tester

        # test不需要sched和optim
        if "sched" in cfg:
            cfg.pop("sched")
        if "optim" in cfg:
            cfg.pop("optim")

        if "train" in cfg["task"]["datasets"]:
            cfg["task"]["datasets"].pop("train")
        if "dev" in cfg["task"]["datasets"]:
            cfg["task"]["datasets"].pop("dev")
        # "test" is required
        assert "test" in cfg["task"]["datasets"]

        cfg["model"]["model_path"] = str(run_result.record_dir / "checkpt" / "best")

        # 因为下面这部分不通用，所以train-and-test不考虑作为一个alchemy的官方入口
        for plugin in cfg["plugins"]:
            if plugin["type"] == "alchemy.plugins.FileLogger":
                train_log_dir = Path(plugin["log_dir"])
                test_log_dir = train_log_dir.with_name("test")
                plugin["log_dir"] = str(test_log_dir)

        evalpipes = cfg["task"]["evalpipes"]
        for evalpipe in list(evalpipes):        # 浅拷贝一份，防止remove in iter
            if evalpipe["type"] == "src.task.ner.evalpipe.EvalNer":
                evalpipe["store_history"] = False
            elif evalpipe["type"] == "src.task.ner.evalpipe.LogBest":
                # 保留log best，可以打印best_info
                pass
            elif evalpipe["type"] == "src.task.ner.evalpipe.SaveStepExamples":
                evalpipe["type"] = "src.task.ner.evalpipe.SaveExamples"
            else:
                evalpipes.remove(evalpipe)

        test_cfgs.append(cfg)

    return run(
        cfgs=test_cfgs,
        device=device,
        user_dir=user_dir,
        desc=desc,
        debug=debug,
        task_per_device=task_per_device,
    )


def train_and_test_entry(
    cfgs: List[str],
    device: Optional[List[int]] = None,
    user_dir: str = "src",
    desc: str = "",
    debug: bool = False,
):
    train_and_test(
        cfgs=[prepare_cfg(Path(cfg)) for cfg in cfgs],
        device=device,
        user_dir=user_dir,
        desc=desc,
        debug=debug,
    )


if __name__ == "__main__":
    typer.run(train_and_test_entry)
