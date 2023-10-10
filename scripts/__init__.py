import json
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Tuple, Union
from loguru import logger
from rich.progress import track
import pandas as pd
from rich.console import Console
from rich.logging import RichHandler


def prepare_logging(tag: Optional[str] = None, level="INFO"):
    console = Console()
    logger.remove()  # remove all handlers, including default handler
    logger.add(
        RichHandler(markup=True, console=console),
        level=level,
        # rich handler已经自带了时间、level和代码所在行
        format="[bold blue]"+ tag + "[/] - {message}" if tag is not None else "{message}",
    )
    return console


class RecIterable(Iterable):
    def __init__(self, xs: Iterable, ys: Iterable) -> None:
        super().__init__()
        assert isinstance(xs, Iterable)
        assert isinstance(ys, Iterable)
        self.xs = xs
        self.ys = ys

    def __iter__(self):
        for x in self.xs:
            for y in self.ys:
                yield (x, ) + (y if isinstance(y, tuple) else (y, ))


def flat_rec_iter(*iters: Iterable):
    """deprecated 请直接使用笛卡尔积

    Returns:
        _type_: _description_
    """
    ret = None
    for iter in reversed(iters):
        if ret is None:
            ret = iter
        else:
            ret = RecIterable(iter, ret)
    return ret
