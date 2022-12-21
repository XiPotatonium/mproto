from rich.console import Console
from typing import Any, Dict, List, Optional, Tuple


def prfs(gt: List[List[Tuple]], pred: List[List[Tuple]], console: Optional[Console] = None, cls_metric=False):
    from sklearn.metrics import precision_recall_fscore_support as prfs
    assert len(gt) == len(pred)
    # import pdb;pdb.set_trace()

    gt_flat = []
    pred_flat = []
    types = set()
    __ty2idx = {"None": 0}
    def ty2idx(ty: str) -> int:
        if ty not in __ty2idx:
            __ty2idx[ty] = len(__ty2idx)
        return __ty2idx[ty]

    for (sample_gt, sample_pred) in zip(gt, pred):
        union = set()
        if cls_metric:
            union.update(sample_gt)
            loc_gt = list(map(lambda x: (x[0], x[1]), sample_gt))
            sample_loc_true_pred = list(filter(lambda x: (x[0], x[1]) in loc_gt, sample_pred))
            union.update(sample_loc_true_pred)
        else:
            union.update(sample_gt)
            union.update(sample_pred)

        for s in union:
            if s in sample_gt:
                t = s[2]
                gt_flat.append(ty2idx(t))
                types.add(t)
            else:
                gt_flat.append(0)       # None

            if s in sample_pred:
                t = s[2]
                pred_flat.append(ty2idx(t))
                types.add(t)
            else:
                pred_flat.append(0)     # None

    types = list(types)
    labels = [ty2idx(t) for t in types]
    per_type = prfs(gt_flat, pred_flat, labels=labels, average=None, zero_division=1)
    micro = prfs(gt_flat, pred_flat, labels=labels, average='micro', zero_division=1)[:-1]
    macro = prfs(gt_flat, pred_flat, labels=labels, average='macro', zero_division=1)[:-1]
    total_support = sum(per_type[-1])

    if console is not None:
        __print_results(per_type, list(micro) + [total_support], list(macro) + [total_support], types, console)

    return [m * 100 for m in micro + macro]


def __print_results(per_type: List, micro: List, macro: List, types: List[str], console: Console):
    columns = ('type', 'precision', 'recall', 'f1-score', 'support')

    row_fmt = "%20s" + (" %12s" * (len(columns) - 1))
    console.print(row_fmt % columns)

    metrics_per_type = []
    for i, t in enumerate(types):
        metrics = []
        for j in range(len(per_type)):
            metrics.append(per_type[j][i])
        metrics_per_type.append(metrics)

    def get_row(data, label):
        row = [label]
        for i in range(len(data) - 1):
            row.append("%.2f" % (data[i] * 100))
        row.append(data[3])
        return tuple(row)

    for m, t in zip(metrics_per_type, types):
        console.print(row_fmt % get_row(m, t))

    console.print('')
    # micro
    console.print(row_fmt % get_row(micro, 'micro'))
    # macro
    console.print(row_fmt % get_row(macro, 'macro'))


def filter_overlapping(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def _is_overlapping(e1, entities):
        def check_overlap(e1, e2):
            if e1["end"] < e2["start"] or e2["end"] < e1["start"]:
                return False
            else:
                return True

        for e2 in entities:
            if check_overlap(e1, e2):
                return True
        return False

    matches = list(matches)                         # 拷贝一份，反正都是引用不浪费什么时间，inplace的sort可能会出现问题
    matches.sort(key=lambda p: p["end"] - p["start"], reverse=True)

    ret = []
    for i, match in enumerate(matches):
        if not _is_overlapping(match, ret):
            ret.append(match)

    return ret


def filter_partial_overlapping(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def _is_partial_overlapping(e1, entities):
        def check_partial_overlap(e1, e2):
            if (
                (e1["start"] < e2["start"] <= e1["end"] < e2["end"]) or
                (e2["start"] < e1["start"] <= e2["end"] < e1["end"])
            ):
                return True
            else:
                return False
        for e2 in entities:
            if check_partial_overlap(e1, e2):
                return True
        return False

    matches = list(matches)                         # 拷贝一份，反正都是引用不浪费什么时间，inplace的sort可能会出现问题
    matches.sort(key=lambda p: p["end"] - p["start"], reverse=True)

    ret = []
    for i, match in enumerate(matches):
        if not _is_partial_overlapping(match, ret):
            ret.append(match)

    return ret
