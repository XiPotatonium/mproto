from typing import Any, Dict, Iterable, List, Tuple
from sklearn.metrics import precision_recall_fscore_support as prfs

from .entities import EntityType


class NerScorer:
    def __init__(self, entities: Dict[str, EntityType]):
        self.entities = dict(entities)
        self._pseudo_entity_type = EntityType('Entity', len(self.entities), 'Entity', 'Entity')  # for span only evaluation
        self.entities["Entity"] = self._pseudo_entity_type

    def compute_scores(self, preds: List[Dict[str, Any]]):
        gt, pred = self.__convert(preds, include_entity_types=True)
        ner_eval = self.__score(gt, pred)

        gt_wo_type, pred_wo_type = self.__convert(preds, include_entity_types=False)
        ner_loc_eval = self.__score(gt_wo_type, pred_wo_type)

        ner_cls_eval = self.__score(gt, pred, cls_metric=True)

        return ner_eval, ner_loc_eval, ner_cls_eval

    def compute_one_sample(self, preds: Dict[str, Any], include_entity_types: bool):
        gt, pred = self.__convert([preds], include_entity_types=include_entity_types, include_score=True)
        gt, pred = gt[0], pred[0]

        # get micro precision/recall/f1 scores
        if gt or pred:
            pred_s = [p[:3] for p in pred]  # remove score
            result = self.__score([gt], [pred_s])
            precision = result["p_micro"]
            recall = result["r_micro"]
            f1 = result["f1_micro"]
        else:
            # corner case: no ground truth and no predictions
            precision, recall, f1 = [100.0] * 3
        return gt, pred, precision, recall, f1

    def __convert(
        self, preds: List[Dict[str, Any]],
        include_entity_types: bool = True, include_score: bool = False
    ):
        converted_gt, converted_pred = [], []

        for sample_preds in preds:
            preds = []
            gts = []
            for pred in sample_preds["preds"]:
                p_tup = [
                    pred["start"],
                    pred["end"],
                    self.entities[pred["type"]] if include_entity_types else self._pseudo_entity_type
                ]

                if include_score:
                    # include prediction scores
                    # tagging这类的方法可能会没有分数，就赋一个1好了
                    p_tup.append(pred.get("type_score", 1.0))
                preds.append(tuple(p_tup))
            for gt in sample_preds["gts"]:
                gt_tup = [
                    gt["start"],
                    gt["end"],
                    self.entities[gt["type"]] if include_entity_types else self._pseudo_entity_type
                ]
                gts.append(tuple(gt_tup))

            converted_gt.append(gts)
            converted_pred.append(preds)

        return converted_gt, converted_pred

    def __score(self, gt: List[List[Tuple]], pred: List[List[Tuple]], cls_metric=False):
        assert len(gt) == len(pred)

        gt_flat = []
        pred_flat = []
        types = set()

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
                    gt_flat.append(t.index)
                    types.add(t)
                else:
                    gt_flat.append(0)

                # 是从union中取出来的，所以重复的预测不影响评估结果
                if s in sample_pred:
                    t = s[2]
                    pred_flat.append(t.index)
                    types.add(t)
                else:
                    pred_flat.append(0)

        labels = [t.index for t in types]
        p, r, f1, support = prfs(gt_flat, pred_flat, labels=labels, average=None, zero_division=1)
        p_micro, r_micro, f1_micro, _ = prfs(gt_flat, pred_flat, labels=labels, average='micro', zero_division=1)
        p_macro, r_macro, f1_macro, _ = prfs(gt_flat, pred_flat, labels=labels, average='macro', zero_division=1)

        return {
            "p": p * 100, "r": r * 100, "f1": f1 * 100, "support": support,
            "p_micro": p_micro * 100, "r_micro": r_micro * 100, "f1_micro": f1_micro * 100,
            "p_macro": p_macro * 100, "r_macro": r_macro * 100, "f1_macro": f1_macro * 100,
            "types": list(types),       # per type里面的类型顺序和这个types一致
        }
