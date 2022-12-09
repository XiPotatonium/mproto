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


def count_items(items: Dict[str, List[str]]) -> int:
    return sum(map(lambda x: len(x), items.values()))


def load_corpus(corpus_dir: Path) -> list:
    data = []
    if not corpus_dir.exists():
        return data
    files = list(corpus_dir.iterdir())
    files.sort()
    for file in track(files):
        with file.open('r', encoding="utf8") as f:
            data.append(json.load(f))
    return data


def dump_corpus(corpus_dir: Path, data: Iterable):
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for i, item in enumerate(track(data)):
        with (corpus_dir / "{}.json".format(i)).open('w', encoding="utf8") as f:
            json.dump(item, f, ensure_ascii=False)


def read_human_anno(anno_path: Path, sheets: List[str]):
    """Return phrase -> label

    Args:
        anno_path (Path): _description_
        sheets (List[str]): _description_

    Returns:
        _type_: _description_
    """
    items = {}
    for sheet in sheets:
        anno = pd.read_excel(anno_path, sheet_name=sheet)
        for row_i in range(len(anno)):
            label = str(anno.iloc[row_i, 0])        # 有可能是nan，所以全部转换为str
            phrase = anno.iloc[row_i, 1]
            if phrase in items and items[phrase] != label:
                logger.warning("Conflict label {} and {} for {}".format(items[phrase], label, phrase))
            items[phrase] = label
    return items


def zh_join(tokens: List[str], with_mapping: bool = False) -> Union[str, Tuple[str, List[str]]]:
    """这里是为了解决一个矛盾，如果分词的时候句子是中文，那么join的时候不需要添加空格，如果两侧是英文，那么需要空格分开

    Args:
        tokens (List[str]): _description_
        with_mapping (bool): default to False

    Returns:
        str: _description_
    """
    puct_rule = "\·\~\！\@\#\￥\%\……\&\*\（\）\——\-\+\=\【\】\{\}\、\|\；\‘\’\：\“\”\《\》\？\，\。\、\`\~\!\#\$\%\^\&\*\(\)\_\[\]{\}\\\|\;\'\'\:\"\"\,\.\/\<\>\?"
    zh_rule = "\u4e00-\u9fa5"
    rule = f"[{puct_rule}{zh_rule}]"

    ret = ''
    mapping = []
    for i, t in enumerate(tokens):
        mapping.append(len(ret))
        ret += t
        # 中文和标点符号不需要空格，因此两侧都不是中文和标点的情况下需要添加空格
        if (
            not re.fullmatch(rule, t[-1]) and
            (i + 1 < len(tokens) and not re.fullmatch(rule, tokens[i + 1][0]))
        ):
            ret += ' '
    if with_mapping:
        return ret, mapping
    else:
        return ret


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
