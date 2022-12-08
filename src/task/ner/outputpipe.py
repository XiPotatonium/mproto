from abc import ABC, abstractmethod
from typing import Any, Dict, List, MutableMapping

from alchemy import sym_tbl
from alchemy.pipeline import OutputPipeline
from . import NerTask
from .entities import EntityType, get_span_tokens


@OutputPipeline.register("PrunePreds")
class PrunePreds(OutputPipeline):
    def __init__(self, preserve: List[str], **kwargs):
        super().__init__()
        self.preserve = set(preserve)

    def __call__(self, outputs: Any, inputs: MutableMapping[str, Any]) -> Any:
        ret = []
        for sample_output in outputs:
            sample_output = dict(sample_output)
            sample_output["preds"] = [
                {
                    k: v for k, v in pred.items()
                    if k in self.preserve
                } for pred in sample_output["preds"]
            ]
            ret.append(sample_output)
        return ret


@OutputPipeline.register("WithSampleInfo")
class WithSampleInfo(OutputPipeline):
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, outputs: Any, inputs: MutableMapping[str, Any]) -> Any:
        task: NerTask = sym_tbl().task
        idx2entity_type: Dict[int, EntityType] = task.idx2entity_type

        def mapper(output: Dict[str, Any]):
            new_output = {}
            for k, v in output.items():
                if k == "start" or k == "end":
                    new_output[k] = v
                elif k == "type":
                    new_output[k] = idx2entity_type[v].identifier
                else:
                    new_output[k] = v
            return new_output

        ret = []
        for sample_outputs, raw_sample in zip(outputs, inputs["raw_sample"]):
            gt_converted_entities = []
            for mention in raw_sample.mentions:
                mention = mention.as_tuple_token()
                mention_span = mention[:2]
                span_tokens = get_span_tokens(raw_sample.tokens, mention_span)
                entity_type = mention[2].identifier
                gt_converted_entities.append({
                    "type": entity_type,
                    "start": span_tokens[0].index,
                    "end": span_tokens[-1].index,       # exclusive
                })
            gt_converted_entities = sorted(gt_converted_entities, key=lambda e: e['start'])
            ret.append({
                "tokens": [t.phrase for t in raw_sample.tokens],
                "preds": list(map(mapper, sample_outputs)),
                "gts": gt_converted_entities,
                "id": raw_sample.id,
            })

        return ret


@OutputPipeline.register("PruneNone")
class PruneNone(OutputPipeline):
    def __init__(self, none_tag: str = "None", **kwargs):
        super().__init__()
        self.none_tag = none_tag

    def __call__(self, outputs: Any, inputs: MutableMapping[str, Any]) -> Any:
        ret = []
        for sample_outputs in outputs:
            sample_outputs = dict(sample_outputs)       # 拷贝一份，不要做inplace的修改
            sample_outputs["preds"] = list(filter(lambda o: o["type"] != self.none_tag, sample_outputs["preds"]))
            ret.append(sample_outputs)
        return ret


@OutputPipeline.register("PruneInvalidSpan")
class PruneInvalidSpan(OutputPipeline):
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, outputs: Any, inputs: MutableMapping[str, Any]) -> Any:
        ret = []
        for sample_outputs in outputs:
            sample_outputs = dict(sample_outputs)       # 拷贝一份，不要做inplace的修改
            sample_outputs["preds"] = list(filter(lambda o: o["start"] <= o["end"], sample_outputs["preds"]))
            ret.append(sample_outputs)
        return ret


@OutputPipeline.register("PruneDuplicate")
class PruneDuplicate(OutputPipeline):
    def __init__(self, weight: Dict[str, float], **kwargs):
        super().__init__()
        self.weight = weight

    def scorer(self, pred: Dict[str, Any]) -> float:
        return sum(w * pred[k] for k, w in self.weight.items())

    def __call__(self, outputs: Any, inputs: MutableMapping[str, Any]) -> Any:
        ret = []

        for sample_outputs in outputs:
            sample_outputs = dict(sample_outputs)       # 拷贝一份，不要做inplace的修改
            preds = sample_outputs["preds"]
            preds = list(preds)                         # 拷贝一份，反正都是引用不浪费什么时间，inplace的sort可能会出现问题

            preds.sort(key=lambda x: self.scorer(x), reverse=True)

            filtered_preds = []
            for i, pred in enumerate(preds):
                if not PruneDuplicate._is_duplicate(pred, filtered_preds):
                    filtered_preds.append(pred)
            sample_outputs["preds"] = filtered_preds
            ret.append(sample_outputs)

        return ret

    @staticmethod
    def _is_duplicate(e1, entities) -> bool:
        for entity in entities:
            if e1["start"] == entity["start"] and e1["end"] == entity["end"]:
                return True
        return False


class FilterOverlapping(ABC, OutputPipeline):

    def __call__(self, outputs: Any, inputs: MutableMapping[str, Any]) -> Any:
        ret = []

        for sample_outputs in outputs:
            sample_outputs = dict(sample_outputs)       # 拷贝一份，不要做inplace的修改
            preds = sample_outputs["preds"]
            preds = list(preds)                         # 拷贝一份，反正都是引用不浪费什么时间，inplace的sort可能会出现问题

            preds.sort(key=lambda x: self.scorer(x), reverse=True)

            filtered_preds = []
            for i, pred in enumerate(preds):
                if not FilterOverlapping._is_overlapping(pred, filtered_preds):
                    filtered_preds.append(pred)
            sample_outputs["preds"] = filtered_preds
            ret.append(sample_outputs)

        return ret

    @abstractmethod
    def scorer(self, pred: Dict[str, Any]) -> float:
        raise NotImplementedError()

    @staticmethod
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


@OutputPipeline.register("PruneOverlappingByConfidence")
class PruneOverlappingByConfidence(FilterOverlapping):
    def __init__(self, weight: Dict[str, float], **kwargs):
        super().__init__()
        self.weight = weight

    def scorer(self, pred: Dict[str, Any]) -> float:
        return sum(w * pred[k] for k, w in self.weight.items())


@OutputPipeline.register("FilterOverlappingByLen")
class FilterOverlappingByLen(FilterOverlapping):
    def __init__(self, **kwargs):
        super().__init__()

    def scorer(self, pred: Dict[str, Any]) -> float:
        return pred["end"] - pred["start"]


class FilterPartialOverlapping(ABC, OutputPipeline):

    def __call__(self, outputs: Any, inputs: MutableMapping[str, Any]) -> Any:
        ret = []

        for sample_outputs in outputs:
            sample_outputs = dict(sample_outputs)       # 拷贝一份，不要做inplace的修改
            preds = sample_outputs["preds"]
            preds = list(preds)                         # 拷贝一份，反正都是引用不浪费什么时间，inplace的sort可能会出现问题

            preds.sort(key=lambda e: self.scorer(e), reverse=True)

            filtered_preds = []
            for i, pred in enumerate(preds):
                if not FilterPartialOverlapping._is_partial_overlapping(pred, filtered_preds):
                    filtered_preds.append(pred)
            sample_outputs["preds"] = filtered_preds
            ret.append(sample_outputs)

        return ret

    @abstractmethod
    def scorer(self, pred: Dict[str, Any]) -> float:
        raise NotImplementedError()

    @staticmethod
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


@OutputPipeline.register("PrunePartialOverlappingByConfidence")
class PrunePartialOverlappingByConfidence(FilterPartialOverlapping):
    def __init__(self, weight: Dict[str, float], **kwargs):
        super().__init__()
        self.weight = weight

    def scorer(self, pred: Dict[str, Any]) -> float:
        return sum(w * pred[k] for k, w in self.weight.items())


@OutputPipeline.register("FilterPartialOverlappingByLen")
class FilterPartialOverlappingByLen(FilterPartialOverlapping):
    def __init__(self, **kwargs):
        super().__init__()

    def scorer(self, pred: Dict[str, Any]) -> float:
        return pred["end"] - pred["start"]


@OutputPipeline.register("PruneByClsScore")
class PruneByClsScore(OutputPipeline):
    def __init__(self, threshold: float, **kwargs):
        super().__init__()
        self.threshold = threshold

    def __call__(self, outputs: Any, inputs: MutableMapping[str, Any]) -> Any:
        ret = []
        for sample_outputs in outputs:
            sample_outputs = dict(sample_outputs)       # 拷贝一份，不要做inplace的修改
            sample_outputs["preds"] = list(filter(
                lambda e: e["type_score"] >= self.threshold, sample_outputs["preds"]
            ))
            ret.append(sample_outputs)
        return ret


@OutputPipeline.register("PruneByBoundaryScore")
class PruneByBoundaryScore(OutputPipeline):
    def __init__(self, threshold: float, **kwargs):
        super().__init__()
        self.threshold = threshold

    def __call__(self, outputs: Any, inputs: MutableMapping[str, Any]) -> Any:
        ret = []
        for sample_outputs in outputs:
            sample_outputs = dict(sample_outputs)       # 拷贝一份，不要做inplace的修改
            sample_outputs["preds"] = list(filter(
                lambda e: e["start_score"] >= self.threshold and e["end_score"] >= self.threshold,
                sample_outputs["preds"]
            ))
            ret.append(sample_outputs)
        return ret
