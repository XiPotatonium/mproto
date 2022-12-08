import copy
from pathlib import Path
from typing import Any, Dict, MutableMapping, Optional
from loguru import logger
import torch
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter
from alchemy import sym_tbl
from alchemy.pipeline import SchedPipeline, BeginStepPipeline, EndStepPipeline
import einops


@SchedPipeline.register("LogLRESPipeline")
class LogLRESPipeline(EndStepPipeline):
    def __init__(
        self,
        log_tensorboard: bool = False,
        varname: str = "summary_writer",
        tag: str = "train/lr",
        log_file: bool = False,
        filename: str = "train_lr.log",
        **kwargs
    ) -> None:
        super().__init__()
        self.log_tensorboard = log_tensorboard
        self.varname = varname
        self.tag = tag
        self.log_file = log_file
        self.filename = filename

    def __call__(self, outputs: MutableMapping[str, Any], **kwargs) -> Dict[str, Any]:
        lrs = sym_tbl().optim.get_lr()

        summary_writer: Optional[SummaryWriter] = sym_tbl().try_get_global(self.varname)
        if self.log_tensorboard and summary_writer is not None:
            summary_writer.add_scalar(
                self.tag, lrs[-1], sym_tbl().train_sched.cur_step
            )
        record_dir: Optional[Path] = sym_tbl().try_get_global("record_dir")
        if self.log_file and record_dir is not None:
            with (record_dir / self.filename).open('a', encoding="utf8") as f:
                f.write("{}\n".format(",".join(lrs)))
        return kwargs


@SchedPipeline.register("LogTrainLossESPipeline")
class LogTrainLossESPipeline(EndStepPipeline):
    def __init__(
        self,
        log_tensorboard: bool = False,
        varname: str = "summary_writer",
        tag: str = "train/loss",
        log_file: bool = False,
        filename: str = "train_loss.log",
        **kwargs
    ) -> None:
        super().__init__()
        self.log_tensorboard = log_tensorboard
        self.varname = varname
        self.tag = tag
        self.log_file = log_file
        self.filename = filename

    def __call__(self, outputs: MutableMapping[str, Any], **kwargs) -> Dict[str, Any]:
        loss = outputs["loss"]

        summary_writer: Optional[SummaryWriter] = sym_tbl().try_get_global(self.varname)
        if self.log_tensorboard and summary_writer is not None:
            summary_writer.add_scalar(
                self.tag, loss, sym_tbl().train_sched.cur_step
            )

        record_dir: Optional[Path] = sym_tbl().try_get_global("record_dir")
        if self.log_file and record_dir is not None:
            with (record_dir / self.filename).open('a', encoding="utf8") as f:
                f.write("{}\n".format(loss))
        return kwargs


@SchedPipeline.register("ModifyConfigBSPipeline")
class ModifyConfigBSPipeline(BeginStepPipeline):
    def __init__(self, step: int, **kwargs) -> None:
        super().__init__()
        self.step = step
        self.kwargs = kwargs

    def __call__(self, batch: MutableMapping[str, Any], **kwargs) -> Dict[str, Any]:
        if sym_tbl().train_sched.cur_step == self.step:
            old_cfg = {}
            for key, value in self.kwargs.items():
                old_cfg[key] = sym_tbl().cfg.get(key)
                sym_tbl().cfg[key] = value
            if len(old_cfg) != 0:
                logger.info("Modify cfg {{{}}} at stage {}".format(
                    ", ".join("{}: {} -> {}".format(k, v, self.kwargs[k]) for k, v in old_cfg.items()),
                    sym_tbl().train_sched.cur_step
                ))
        return kwargs


@SchedPipeline.register("ResetOptimSchedBSPipeline")
class ResetOptimSchedBSPipeline(BeginStepPipeline):
    def __init__(self, step: int, **kwargs) -> None:
        super().__init__()
        self.step = step
        self.kwargs = kwargs

    def __call__(self, batch: MutableMapping[str, Any], **kwargs) -> Dict[str, Any]:
        if sym_tbl().train_sched.cur_step == self.step:
            logger.info(
                "Reset optimizer and lr_sched with cfg {} at step {}".format(
                    self.kwargs, sym_tbl().train_sched.cur_step
                )
            )

            sym_tbl().optim.reset(**self.kwargs)
            sym_tbl().train_sched.reset_lr_sched(**self.kwargs)

            if sym_tbl().device.type != "cpu":
                with torch.cuda.device(sym_tbl().device):
                    # empty cache uses GPU 0 by default
                    torch.cuda.empty_cache()
        return kwargs


@SchedPipeline.register()
class BondSLForTaggingBSPipeline(BeginStepPipeline):
    def __init__(
        self,
        begin_step: int,
        period: int,
        label_selection_threshold: float,
        use_ema: bool = False,
        ema_rate: float = 0.999,
        **kwargs
    ) -> None:
        super().__init__()
        self.teacher: Module = None
        self.begin_step = begin_step
        self.period = period
        self.label_selection_threshold = label_selection_threshold
        self.use_ema = use_ema
        self.ema_rate = ema_rate

    @staticmethod
    def ema_update(model: Module, current: Module, decay: float):
        model_param_map = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                model_param_map[name] = param
        for name, param in current.named_parameters():
            if param.requires_grad:
                assert name in model_param_map
                new_average = (1.0 - decay) * param.data + decay * model_param_map[name].data
                model_param_map[name].data = new_average.detach()

    @staticmethod
    def soft_frequency(logits, power=2, probs=False):
        """
        Unsupervised Deep Embedding for Clustering Analysis
        https://arxiv.org/abs/1511.06335
        """
        if not probs:
            softmax = torch.nn.Softmax(dim=1)
            y = softmax(logits.view(-1, logits.shape[-1])).view(logits.shape)
        else:
            y = logits
        f = torch.sum(y, dim=(0, 1))
        t = y**power / f
        p = t/torch.sum(t, dim=2, keepdim=True)

        return p

    def __call__(self, batch: MutableMapping[str, Any], **kwargs) -> Dict[str, Any]:

        step = sym_tbl().train_sched.cur_step
        if step >= self.begin_step:
            # Update a new teacher periodically
            delta = step - self.begin_step
            if delta % self.period == 0:
                logger.info("Updating teacher at step {}...".format(step))
                # 注意AlchemyModel不能deepcopy，里面存在ctx
                if self.teacher is None or not self.use_ema:
                    self.teacher = copy.deepcopy(sym_tbl().model.model)
                else:
                    self.ema_update(self.teacher, copy.deepcopy(sym_tbl().model.model), self.ema_rate)
                self.teacher.eval()
                self.teacher.to(sym_tbl().device)

            # Using current teacher to update the label
            with torch.no_grad():
                student_model = sym_tbl().model.model
                sym_tbl().model.model = self.teacher
                outputs = sym_tbl().model.forward(
                    batch,
                    needs_loss=False,
                    requires_grad=False
                )
                sym_tbl().model.model = student_model

            # if args.self_training_label_mode == "hard":
            #     pred_labels = torch.argmax(outputs[0], axis=2)
            #     pred_labels, label_mask = multi_source_label_refine(args, batch[5], batch[3], pred_labels, pad_token_label_id, pred_logits=outputs[0])
            # elif args.self_training_label_mode == "soft":
            pred_labels = self.soft_frequency(logits=outputs["logits"], power=2)
            # pred_labels = torch.softmax(outputs, dim=-1)
            # NOTE: 对于原型网络而言，一个很有可能发生的问题是一个表征和多个原型接近（这在这几个原型本身就很接近的情况下是很常见的）
            # 这使得我们不能直接依赖最大置信度来过滤，可能需要先按照类型进行一个add，然后再过滤
            label_masks = (pred_labels.max(dim=-1)[0] > self.label_selection_threshold)
            batch["label_masks"] = label_masks.cpu()
            # 我们没有高精度label的说法，所有的label都是一样的
            batch["soft_gt_seq_labels"] = pred_labels.detach().cpu()
        return kwargs

@SchedPipeline.register()
class BondSLForProtoBSPipeline(BondSLForTaggingBSPipeline):
    def __call__(self, batch: MutableMapping[str, Any], **kwargs) -> Dict[str, Any]:
        step = sym_tbl().train_sched.cur_step
        if step >= self.begin_step:
            # Update a new teacher periodically
            delta = step - self.begin_step
            if delta % self.period == 0:
                logger.info("Updating teacher at step {}...".format(step))
                # 注意AlchemyModel不能deepcopy，里面存在ctx
                if self.teacher is None or not self.use_ema:
                    self.teacher = copy.deepcopy(sym_tbl().model.model)
                else:
                    self.ema_update(self.teacher, copy.deepcopy(sym_tbl().model.model), self.ema_rate)
                self.teacher.eval()
                self.teacher.to(sym_tbl().device)

            # Using current teacher to update the label
            with torch.no_grad():
                student_model = sym_tbl().model.model
                sym_tbl().model.model = self.teacher
                outputs = sym_tbl().model.forward(
                    batch,
                    needs_loss=False,
                    requires_grad=False
                )
                sym_tbl().model.model = student_model

            pred_labels = self.soft_frequency(logits=outputs["logits"], power=2)
            # 对于原型网络而言，一个很有可能发生的问题是一个表征和多个原型接近（这在这几个原型本身就很接近的情况下是很常见的）
            # 这使得我们不能直接依赖最大置信度来过滤，可能需要先按照类型进行一个add，然后再过滤
            pred_labels_k = torch.sum(
                einops.rearrange(pred_labels, "b s (k p) -> b s k p", p=sym_tbl().model.num_proto_per_type),
                dim=-1
            )
            label_masks = (pred_labels_k.max(dim=-1)[0] > self.label_selection_threshold)
            batch["label_masks"] = label_masks.cpu()
            batch["soft_gt_seq_labels"] = pred_labels.detach().cpu()
            batch["soft_gt_seq_logits"] = outputs["logits"].detach().cpu()
        return kwargs
