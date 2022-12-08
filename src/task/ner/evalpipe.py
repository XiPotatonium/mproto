import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from loguru import logger

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from transformers import PreTrainedModel
from alchemy import AlchemyTrainScheduler, sym_tbl
from alchemy.pipeline import EvalPipeline
from alchemy.util.json import NpJsonEncoder
from .entities import EntityType
from .ner_scorer import NerScorer
from . import NerTask


@EvalPipeline.register("EvalNer")
class EvalNerPipe(EvalPipeline):
    def __init__(
        self,
        varname: str,
        print_results: bool = True,
        store_history: bool = True,
        **kwargs
    ) -> None:
        super().__init__()
        self.print_results = print_results
        self.varname = varname
        # test可能不需要store_history
        self.store_history = store_history

    def __call__(self, split: str, **kwargs) -> Dict[str, Any]:
        preds = sym_tbl().pop_global(self.varname)
        kwargs["preds"] = preds

        # 计算评估分数
        task: NerTask = sym_tbl().task
        ner_score, loc_eval_score, cls_eval_score = task.scorer.compute_scores(preds)
        eval_score = {
            "ner_score": ner_score,
            "loc_score": loc_eval_score,
            "cls_score": cls_eval_score,
        }

        save_dir: Optional[Path] = sym_tbl().try_get_global("record_dir")
        if self.store_history and save_dir is not None:
            self.save_eval_history(save_dir / "evals.jsonl", eval_score)

        kwargs.update(eval_score)

        if self.print_results:
            logger.info("--- NER ---")
            logger.info("")
            self.__print_results(ner_score)

            logger.info("")
            logger.info("--- NER on Localization ---")
            logger.info("")
            self.__print_results(loc_eval_score)

            logger.info("")
            logger.info("--- NER on Classification ---")
            logger.info("")
            self.__print_results(cls_eval_score)

        return kwargs

    @staticmethod
    def save_eval_history(path: Path, score: Dict[str, Any]):
        """types不能序列化，这里做一个转换

        Args:
            path (Path): _description_
            score (Dict[str, Any]): _description_
        """
        def map_types(score_: Dict[str, Any]):
            new_score = {}
            for k, v in score_.items():
                if k == "types":
                    v = [t.identifier for t in v]
                new_score[k] = v
            return new_score

        score_for_dump = {k: map_types(s) for k, s in score.items()}

        with path.open('a', encoding="utf8") as wf:
            wf.write(json.dumps({
                "step": sym_tbl().train_sched.cur_step,
                "score": score_for_dump,
            }, cls=NpJsonEncoder))
            wf.write('\n')
        pass

    @staticmethod
    def __print_results(eval_result: Dict[str, Any]):
        per_type = [eval_result["p"], eval_result["r"], eval_result["f1"], eval_result["support"]]
        total_support = sum(eval_result["support"])
        micro = [eval_result["p_micro"], eval_result["r_micro"], eval_result["f1_micro"], total_support]
        macro = [eval_result["p_macro"], eval_result["r_macro"], eval_result["f1_macro"], total_support]
        types: List[EntityType] = eval_result["types"]

        columns = ('type', 'precision', 'recall', 'f1-score', 'support')

        row_fmt = "%20s" + (" %12s" * (len(columns) - 1))
        logger.info(row_fmt % columns)

        metrics_per_type = []
        for i, t in enumerate(types):
            metrics = []
            for j in range(len(per_type)):
                metrics.append(per_type[j][i])
            metrics_per_type.append(metrics)

        def get_row(data, label):
            row = [label]
            for i in range(len(data) - 1):
                row.append("%.2f" % (data[i]))
            row.append(data[3])
            return tuple(row)

        for m, t in zip(metrics_per_type, types):
            logger.info(row_fmt % get_row(m, t.short_name))

        logger.info('')
        # micro
        logger.info(row_fmt % get_row(micro, 'micro'))
        # macro
        logger.info(row_fmt % get_row(macro, 'macro'))


@EvalPipeline.register("LogBest")
class LogBest(EvalPipeline):
    def __init__(self, metric: str = "f1_micro", **kwargs):
        super().__init__()
        self.metric = metric
        self.save_dir: Optional[Path] = sym_tbl().try_get_global("record_dir")

    def begin_eval(self, split: str, **kwargs):
        sym_tbl().try_set_global("best_info", {"epoch": -1, "step": -1})

    def __call__(self, split: str, **kwargs) -> Dict[str, Any]:
        best_info = sym_tbl().get_global("best_info")
        if self.metric == "loss":
            score = kwargs["loss"]
            achieve_best = best_info["step"] == -1 or score < best_info[self.metric]
        else:
            score = kwargs["ner_score"][self.metric]
            achieve_best = best_info["step"] == -1 or best_info[self.metric] < score
        kwargs["achieve_best"] = achieve_best

        sched: Optional[AlchemyTrainScheduler] = sym_tbl().train_sched
        cur_epoch = "unk" if sched is None else sched.cur_epoch
        cur_step = "unk" if sched is None else sched.cur_step
        if achieve_best:
            logger.info("Achieve {} = {} at epoch {} (step {}), last best = {}".format(
                self.metric, score, cur_epoch, cur_step, best_info
            ))
            best_info[self.metric] = score
            best_info["epoch"] = cur_epoch
            best_info["step"] = cur_step
            save_dir: Optional[Path] = sym_tbl().try_get_global("record_dir")
            if save_dir is not None:
                with (save_dir / "best_info.json").open('w', encoding="utf8") as wf:
                    json.dump(best_info, wf, ensure_ascii=False)
        else:
            logger.info("Current best = {}".format(best_info))

        return kwargs


@EvalPipeline.register("LogTensorboard")
class LogTensorboard(EvalPipeline):
    def __init__(self, varname: str = "summary_writer", **kwargs) -> None:
        super().__init__()
        self.varname = varname

    def __call__(self, split: str, **kwargs) -> Dict[str, Any]:
        # tensorboard
        sched: AlchemyTrainScheduler = sym_tbl().train_sched
        if sched is None:
            raise Exception(
                "Scheduler is not constructed, "
                f"you may have to specify {AlchemyTrainScheduler.__name__} in config"
            )
        summary_writer: Optional[SummaryWriter] = sym_tbl().try_get_global(self.varname)
        if summary_writer is not None:
            LogTensorboard.log_eval_tensorboard(
                kwargs["ner_score"], kwargs["loc_score"], kwargs["cls_score"],
                summary_writer, sched.cur_step, split
            )

        return kwargs

    @staticmethod
    def log_eval_tensorboard(
        ner_score: Dict[str, Any], loc_score: Dict[str, Any], cls_score: Dict[str, Any],
        summary_writer: SummaryWriter, step: int, label: str,
    ):
        for tag, score in zip(["ner", "loc", "cls"], [ner_score, loc_score, cls_score]):
            for metric in ["p_micro", "r_micro", "f1_micro", "p_macro", "r_macro", "f1_macro"]:
                summary_writer.add_scalar('{}/{}_{}'.format(label, tag, metric), score[metric], step)


@EvalPipeline.register("SaveExamples")
class SaveExamples(EvalPipeline):
    def __init__(
        self,
        template: str,
        save_dir: str,
        max_example_count: int = -1,
        **kwargs
    ):
        super().__init__()
        self.template = Path(template)
        self.max_example_count = max_example_count
        self.save_dir = save_dir

    def __call__(self, split: str, **kwargs) -> Dict[str, Any]:
        preds = kwargs["preds"]
        task: NerTask = sym_tbl().task

        record_dir: Optional[Path] = sym_tbl().try_get_global("record_dir")
        # 保存详细的记录
        if record_dir is not None:
            save_dir = record_dir / self.save_dir
            save_dir.mkdir(exist_ok=True)
            SaveExamples.store_examples(
                store_dir=save_dir,
                scorer=task.scorer,
                all_preds=preds,
                html_template_path=self.template,
                max_example_count=self.max_example_count,
                label=split,
            )

        return kwargs

    @staticmethod
    def store_examples(
        store_dir: Path,
        scorer: NerScorer,
        all_preds: List[Dict[str, Any]],
        html_template_path: Path,
        label: str,
        max_example_count=-1,
    ):
        entity_examples = []

        for i, preds_with_doc in enumerate(all_preds):
            entity_example = SaveExamples.__convert_example(
                scorer,
                preds_with_doc,
                include_entity_types=True
            )
            entity_examples.append(entity_example)

        # entities
        SaveExamples.__store_examples(
            entity_examples[:max_example_count] if max_example_count >= 0 else entity_examples,
            file_path=store_dir / 'examples_entities_{}.html'.format(label),
            template_path=html_template_path
        )

        # __store_examples(
        #     sorted(
        #         entity_examples[:max_example_count] if max_example_count >= 0 else entity_examples,
        #         key=lambda k: k['length']
        #     ),
        #     file_path=os.path.join(
        #         store_dir, 'examples_{}_{}_step_{}.html'.format('entities_sorted', dataset_label, step)
        #     ),
        #     template_path=html_template_path
        # )

    @staticmethod
    def __store_examples(examples: List[Dict], file_path: Path, template_path: Path):
        import jinja2

        # read template
        with template_path.open('r') as f:
            template = jinja2.Template(f.read())

        # write to disc
        template.stream(examples=examples).dump(str(file_path))

    @staticmethod
    def __convert_example(
        scorer: NerScorer,
        preds_with_doc: Dict[str, Any],
        include_entity_types: bool
    ):
        # encoding = doc.encoding
        tokens = preds_with_doc["tokens"]

        gt, preds_with_doc, precision, recall, f1 = scorer.compute_one_sample(preds_with_doc, include_entity_types)

        cls_scores = [p[3] for p in preds_with_doc]
        preds_with_doc = [p[:3] for p in preds_with_doc]
        union = set(gt + preds_with_doc)

        # true positives
        tp = []
        # false negatives
        fn = []
        # false positives
        fp = []

        for s in union:
            type_verbose = s[2].verbose_name

            if s in gt:
                if s in preds_with_doc:
                    cls_score = cls_scores[preds_with_doc.index(s)]
                    tp.append((SaveExamples.__entity_to_html(s, tokens),
                            type_verbose, cls_score))
                else:
                    fn.append((SaveExamples.__entity_to_html(s, tokens), type_verbose, -1))
            else:
                cls_score = cls_scores[preds_with_doc.index(s)]
                fp.append((SaveExamples.__entity_to_html(s, tokens), type_verbose, cls_score))

        tp = sorted(tp, key=lambda p: p[2], reverse=True)
        fp = sorted(fp, key=lambda p: p[2], reverse=True)

        text = " ".join(tokens)

        # text = self._prettify(self._text_encoder.decode(encoding))
        text = SaveExamples.__prettify(text)
        return {
            "text": text,
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "length": len(tokens)
        }

    @staticmethod
    def __prettify(text: str):
        text = text.replace('_start_', '').replace(
            '_classify_', '').replace('<UNK>', '').replace('⁇', '')
        text = text.replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '')
        return text

    @staticmethod
    def __entity_to_html(entity: Tuple, tokens: List[str]):
        start, end = entity[:2]
        entity_type = entity[2].verbose_name

        tag_start = ' <span class="entity">'
        tag_start += '<span class="type">%s</span>' % entity_type

        # ctx_before = self._text_encoder.decode(encoding[:start])
        # e1 = self._text_encoder.decode(encoding[start:end])
        # ctx_after = self._text_encoder.decode(encoding[end:])

        ctx_before = ""
        ctx_after = ""
        e1 = ""
        for i in range(start):
            ctx_before += tokens[i]
            if i != start - 1:
                ctx_before += " "
        for i in range(end + 1, len(tokens)):
            ctx_after += tokens[i]
            if i != (len(tokens) - 1):
                ctx_after += " "
        for i in range(start, end + 1):
            e1 += tokens[i]
            if i != end:
                e1 += " "

        html = ctx_before + tag_start + e1 + '</span> ' + ctx_after
        html = SaveExamples.__prettify(html)

        return html


@EvalPipeline.register("SaveStepExamples")
class SaveStepExamples(SaveExamples):
    def __init__(
        self,
        template: str,
        save_dir: str,
        max_example_count: int = -1,
        **kwargs
    ) -> None:
        super().__init__(template, save_dir, max_example_count, **kwargs)

    def __call__(self, split: str, **kwargs) -> Dict[str, Any]:
        preds = kwargs["preds"]
        task: NerTask = sym_tbl().task

        # 保存详细的记录
        sched: AlchemyTrainScheduler = sym_tbl().train_sched
        if sched is None:
            raise Exception(
                "Scheduler is not constructed, "
                f"you may have to specify {AlchemyTrainScheduler.__name__} in config "
                f"or use {SaveExamples.__name__} instead."
            )
        record_dir: Optional[Path] = sym_tbl().try_get_global("record_dir")
        # 保存详细的记录
        if record_dir is not None:
            save_dir = record_dir / self.save_dir
            save_dir.mkdir(exist_ok=True)
            SaveExamples.store_examples(
                store_dir=save_dir,
                scorer=task.scorer,
                all_preds=preds,
                html_template_path=self.template,
                max_example_count=self.max_example_count,
                label="{}_step{}".format(split, sched.cur_step)
            )

        return kwargs


@EvalPipeline.register("SaveModel")
class SaveModel(EvalPipeline):
    def __init__(self, store_best: bool = False, store_all: bool = False, **kwargs) -> None:
        super().__init__()
        self.store_best = store_best
        self.store_all = store_all

    def __call__(self, split: str, **kwargs) -> Dict[str, Any]:
        checkpt_dir: Optional[Path] = sym_tbl().try_get_global("checkpt_dir")
        achieve_best = kwargs["achieve_best"]

        # 保存checkpoint
        if checkpt_dir is not None:
            if achieve_best and self.store_best:
                save_path = checkpt_dir / "best"
                logger.info("Saving best model at \"{}\"...".format(save_path))
                self._save_checkpt(save_path, sym_tbl().model.model, sym_tbl().task.tokenizer, **kwargs)

            if self.store_all:
                # 如果要求每个eval都保存
                save_path = checkpt_dir / "step{}".format(sym_tbl().train_sched.cur_step)
                logger.info("Saving model at \"{}\"...".format(save_path))
                self._save_checkpt(save_path, sym_tbl().model.model, sym_tbl().task.tokenizer, **kwargs)

        return kwargs

    def _save_checkpt(self, save_dir: Path, model: Union[PreTrainedModel, nn.Module], tokenizer, **kwargs):
        extra = {
            "epoch": sym_tbl().train_sched.cur_epoch, "step": sym_tbl().train_sched.cur_step
        }
        # 暂时不要保存optimizer，太大了
        # if optimizer:
        #     extra['optimizer'] = optimizer.state_dict()

        save_dir.mkdir(parents=True, exist_ok=True)

        # save model
        if isinstance(model, PreTrainedModel):
            model.save_pretrained(str(save_dir))
        else:
            raise NotImplementedError()
        # save vocabulary
        tokenizer.save_pretrained(str(save_dir))
        # save extra
        state_path = save_dir / 'extra.state'
        torch.save(extra, str(state_path))


# from alchemy.util.ddp.accelerate import SaveModelMixin as SaveAccelerateModelMixin
# @EvalPipeline.register("SaveModelAccelerate")
# class SaveModelAccelerate(SaveModel, SaveAccelerateModelMixin):
#     def __init__(self, store_best: bool = False, store_all: bool = False, **kwargs) -> None:
#         super().__init__(store_best, store_all, **kwargs)
#
#     def _save_checkpt(self, save_dir: Path, model: Union[PreTrainedModel, nn.Module], tokenizer, **kwargs):
#         model = self._prepare_to_save(model)
#         return super()._save_checkpt(save_dir, model, tokenizer, **kwargs)
