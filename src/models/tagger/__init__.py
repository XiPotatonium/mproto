from pathlib import Path
from typing import Any, Dict, MutableMapping, Optional
from loguru import logger

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss, KLDivLoss, Embedding
import torch.nn.functional as F
from transformers import BertConfig, RobertaConfig

from alchemy import AlchemyModel, sym_tbl
from alchemy.pipeline import OutputPipeline
from alchemy.util import batch_to_device, filter_optional_cfg
from alchemy.util.optim import prepare_trf_based_model_params
from ...task.ner.tagging import BioTaggingScheme, IOTaggingScheme, TaggingScheme
from ...task.ner import NerTask
from .model import CosineProtoTagger, L2ProtoTagger, Tagger
import einops



@OutputPipeline.register("ProcTaggingOutput")
class ProcTaggingOutput(OutputPipeline):
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, outputs: Any, inputs: MutableMapping[str, Any]) -> Any:
        ret = []
        scheme: TaggingScheme = sym_tbl().model.tagging_scheme

        tags = outputs["tags"]
        token_masks = inputs['token_masks']
        token_count = torch.sum(token_masks.long(), dim=-1).cpu().numpy()

        for sample_i, (pred_tag, tok_cnt) in enumerate(zip(tags, token_count)):
            preds = scheme.decode_tags(pred_tag[: tok_cnt])

            sample_outputs = []
            for pred_left, pred_right, pred_type in preds:
                # NOTE: 注意decoding tags得到的是range而不是span，右边界是exclusive的
                sample_outputs.append(
                    {"start": pred_left, "end": pred_right - 1, "type": pred_type}
                )
            ret.append(sample_outputs)
        return ret


@AlchemyModel.register("Tagger")
class AlchemyTagger(AlchemyModel):
    MODEL_CLASS = Tagger

    def __init__(self):
        super().__init__()
        task: NerTask = sym_tbl().task
        # scheme和模型绑定而不是和task绑定，所以应该放在模型里
        self.tagging_scheme = BioTaggingScheme(task.entity_types)

        plm_type = self.model_cfg["plm_type"]
        if plm_type == "roberta":
            plm_cfg_cls = RobertaConfig
        elif plm_type == "bert":
            plm_cfg_cls = BertConfig
        else:
            raise ValueError("Unimplemented plm_type \"{}\"".format(plm_type))

        if "model_path" in self.model_cfg:
            self.config = plm_cfg_cls.from_pretrained(self.model_cfg["model_path"])
            self.model = self.MODEL_CLASS.from_pretrained(
                self.model_cfg["model_path"],
                config=self.config,
                num_tags=self.tagging_scheme.num_tags,

                # 一些可选的config，即如果文件中没有配置，那么就默认
                **filter_optional_cfg(self.model_cfg, {
                    "dropout",
                })
            )
        else:
            # create model
            self.config = plm_cfg_cls.from_pretrained(self.model_cfg["plm_path"])
            self.model = self.MODEL_CLASS(
                config=self.config,
                num_tags=self.tagging_scheme.num_tags,

                # 一些可选的config，即如果文件中没有配置，那么就默认
                **filter_optional_cfg(self.model_cfg, {
                    "dropout",
                })
            )

            logger.info("Use \"{}\" to initialize {} encoder".format(
                self.model_cfg["plm_path"], self.module.encoder.__class__.__name__
            ))
            self.module.encoder = self.module.encoder.__class__.from_pretrained(self.model_cfg["plm_path"])

    def max_positions(self):
        """和Bert一样

        Returns:
            _type_: _description_
        """
        return self.config.max_position_embeddings

    def set_requires_grad(self, requires_grad: bool, mode: str, **kwargs):
        if mode == "encoder":
            for _, param in self.module.encoder.named_parameters():
                param.requires_grad = requires_grad
        else:
            raise NotImplementedError(mode)

    def optim_params(self, **kwargs):
        return prepare_trf_based_model_params(
            self.model,
            self.module.encoder,
            **filter_optional_cfg(kwargs, {"weight_decay", "trf_lr"}),
        )

    def forward(
        self,
        batch: MutableMapping[str, Any],
        needs_loss: bool,
        requires_grad: bool,
        **kwargs
    ) -> MutableMapping[str, Any]:
        batch = batch_to_device(batch, sym_tbl().device)

        hidden, logits = self.model.forward(
            encodings=batch["encoding"],
            encoding_masks=batch["encoding_masks"],
            token2start=batch["token2start"],
        )
        token_masks = batch['token_masks']

        outputs = {
            "hidden": hidden,
            "logits": logits,
            "tags": torch.argmax(logits, dim=-1).detach().cpu().numpy()
        }

        if needs_loss:
            bp_masks = token_masks.flatten()     # bsz * n_tokens

            if "soft_gt_seq_labels" in batch:
                soft_gts = batch["soft_gt_seq_labels"]
                assert soft_gts.shape == logits.shape

                additional_label_masks = batch.get("label_masks")
                if additional_label_masks is not None:
                    bp_masks = bp_masks & additional_label_masks.flatten()
                flat_logits = logits.flatten(0, 1)[bp_masks]
                flat_soft_gts = soft_gts.flatten(0, 1)[bp_masks]

                # 不知道为什么BOND居然不用log softmax，并且似乎不用更好
                # probs = active_logits
                probs = torch.log_softmax(flat_logits, dim=-1)
                loss = F.kl_div(probs, flat_soft_gts, reduction="batchmean")
            else:
                gts = batch["gt_seq_labels"]
                flat_gts = gts.flatten()[bp_masks]
                flat_logits = logits.flatten(0, 1)[bp_masks]

                # 负采样
                neg_ratio = self.criterion_cfg.get("neg_ratio", -1)
                if neg_ratio < 0:
                    pass
                else:
                    # 做一些随机负采样
                    sample_masks = (flat_gts != 0) | (torch.rand_like(flat_gts, dtype=torch.float) < neg_ratio)
                    flat_logits = torch.masked_select(flat_logits, sample_masks.unsqueeze(-1)).view(-1, logits.size(-1))
                    flat_gts = torch.masked_select(flat_gts, sample_masks)

                if self.criterion_cfg["type"] == "ce":
                    loss_fct = CrossEntropyLoss()
                elif self.criterion_cfg["type"] == "gce":
                    gce_q = self.criterion_cfg.get("q")
                    raise NotImplementedError()
                else:
                    raise NotImplementedError(f"Unknown criterion {self.criterion_cfg}")
                loss = loss_fct.forward(flat_logits, flat_gts)

            outputs["loss"] = self.backward.backward(loss, requires_grad=requires_grad)
        return outputs


@AlchemyModel.register("CosineProtoTagger")
class AlchemyCosineProtoTagger(AlchemyModel):
    MODEL_CLASS = CosineProtoTagger

    def __init__(self):
        super().__init__()
        task: NerTask = sym_tbl().task
        # scheme和模型绑定而不是和task绑定，所以应该放在模型里
        self.tagging_scheme = IOTaggingScheme(task.entity_types)     # 原型应该用IO tagging比较合适

        plm_type = self.model_cfg["plm_type"]
        if plm_type == "roberta":
            plm_cfg_cls = RobertaConfig
        elif plm_type == "bert":
            plm_cfg_cls = BertConfig
        else:
            raise ValueError("Unimplemented plm_type \"{}\"".format(plm_type))

        if "model_path" in self.model_cfg:
            self.config = plm_cfg_cls.from_pretrained(self.model_cfg["model_path"])
        else:
            # create model
            self.config = plm_cfg_cls.from_pretrained(self.model_cfg["plm_path"])

        type_ebd_path = self.model_cfg.get("type_ebd_path")
        if type_ebd_path is None:
            # IO tagging，有多少tag就是多少type
            type_ebd = torch.randn((self.tagging_scheme.num_tags * self.num_proto_per_type, self.config.hidden_size))
        else:
            with Path(type_ebd_path).open('rb') as rf:
                type_ebd: Tensor = torch.load(rf)
                assert (
                    type_ebd.dim() == 3 and
                    type_ebd.size() == (self.tagging_scheme.num_tags, self.num_proto_per_type, self.config.hidden_size)
                ), "Invalid type_ebd ({})".format(type_ebd.size())
                logger.info("Load type ebd ({}) from {}".format(type_ebd.size(), type_ebd_path))
                type_ebd = type_ebd.flatten(0, 1).detach()
        type_ebd.requires_grad_ = True

        if "model_path" in self.model_cfg:
            self.model = self.MODEL_CLASS.from_pretrained(
                self.model_cfg["model_path"],
                config=self.config,
                type_ebd=type_ebd,
                use_learnable_scalar=self.model_cfg["use_learnable_scalar"],

                # 一些可选的config，即如果文件中没有配置，那么就默认
                **filter_optional_cfg(self.model_cfg, {
                    "dropout",
                })
            )
        else:
            # create model
            self.model = self.MODEL_CLASS(
                config=self.config,
                type_ebd=type_ebd,
                use_learnable_scalar=self.model_cfg["use_learnable_scalar"],

                # 一些可选的config，即如果文件中没有配置，那么就默认
                **filter_optional_cfg(self.model_cfg, {
                    "dropout",
                })
            )

            logger.info("Use \"{}\" to initialize {} encoder".format(
                self.model_cfg["plm_path"], self.module.encoder.__class__.__name__
            ))
            self.module.encoder = self.module.encoder.__class__.from_pretrained(self.model_cfg["plm_path"])

    @property
    def num_proto_per_type(self):
        return self.model_cfg.get("num_proto_per_type", 1)

    def max_positions(self):
        """和Bert一样

        Returns:
            _type_: _description_
        """
        return self.config.max_position_embeddings

    def set_requires_grad(self, requires_grad: bool, mode: str, **kwargs):
        if mode == "encoder":
            for _, param in self.module.encoder.named_parameters():
                param.requires_grad = requires_grad
        else:
            raise NotImplementedError(mode)

    def optim_params(self, **kwargs):
        return prepare_trf_based_model_params(
            self.model,
            self.module.encoder,
            **filter_optional_cfg(kwargs, {"weight_decay", "trf_lr"}),
        )

    def forward(
        self, batch: MutableMapping[str, Any], needs_loss: bool, requires_grad: bool, **kwargs
    ) -> MutableMapping[str, Any]:
        batch = batch_to_device(batch, sym_tbl().device)

        hidden, unscaled_logits, logits = self.model.forward(
            encodings=batch["encoding"],
            encoding_masks=batch["encoding_masks"],
            token2start=batch["token2start"],
        )
        token_masks = batch['token_masks']
        pred_tags = torch.div(torch.argmax(logits, dim=-1), self.num_proto_per_type, rounding_mode="floor")

        outputs = {
            "hidden": hidden,
            "unscaled_logits": unscaled_logits,
            "logits": logits,
            "tags": pred_tags.detach().cpu().numpy(),
        }

        if needs_loss:
            losses = []
            loss_weights = []

            bp_masks = token_masks.flatten()     # bsz * n_tokens
            gts = batch["gt_seq_labels"]
            flat_logits = logits.flatten(0, 1)[bp_masks]
            flat_gts = gts.flatten()[bp_masks]

            # 负采样
            neg_ratio = self.criterion_cfg.get("neg_ratio", -1)
            if neg_ratio < 0:
                pass
            else:
                # 做一些随机负采样
                sample_masks = (flat_gts != 0) | (torch.rand_like(flat_gts, dtype=torch.float) < neg_ratio)
                flat_logits = torch.masked_select(flat_logits, sample_masks.unsqueeze(-1)).view(-1, logits.size(-1))
                flat_gts = torch.masked_select(flat_gts, sample_masks)

            if self.criterion_cfg["match_type"] == "nearest":
                match_index = self.nearest_matching(flat_logits, flat_gts)
            elif self.criterion_cfg["match_type"] == "sinkhorn":
                match_index = self.sinkhorn_matching(flat_logits, flat_gts)
            elif self.criterion_cfg["match_type"] == "sinkhorn_pot":
                match_index = self.sinkhorn_pot_matching(flat_logits, flat_gts)
            else:
                raise NotImplementedError("Unimplemented match type {}".format(self.criterion_cfg["match_type"]))
            if self.criterion_cfg["type"] == "ce":
                loss_fct = CrossEntropyLoss()
            elif self.criterion_cfg["type"] == "gce":
                gce_q = self.criterion_cfg.get("q")
                raise NotImplementedError()
            else:
                raise NotImplementedError(f"Unknown criterion {self.criterion_cfg}")
            label_loss = loss_fct.forward(flat_logits, match_index)
            losses.append(label_loss)
            loss_weights.append(1.0)

            if "soft_gt_seq_labels" in batch:
                pseudo_loss, label_weight, pseudo_weight = self.pseudo_loss(
                    soft_gts=batch["soft_gt_seq_labels"],
                    soft_label_masks=batch.get("label_masks"),       # 一般是置信度mask
                    logits=logits,
                    token_masks=bp_masks,
                    label_loss=label_loss.item(),
                )
                loss_weights[-1] = label_weight
                losses.append(pseudo_loss)
                loss_weights.append(pseudo_weight)

            compact_loss, compact_weight = self.compact_loss(flat_logits, match_index)
            losses.append(compact_loss)
            loss_weights.append(compact_weight)

            proto_cl_loss, proto_cl_weight = self.proto_cl_loss()
            losses.append(proto_cl_loss)
            loss_weights.append(proto_cl_weight)

            outputs["loss"] = self.backward.backward(losses, loss_weights, requires_grad=requires_grad)
        return outputs

    def pseudo_loss(
        self,
        soft_gts: Tensor,
        soft_label_masks: Optional[Tensor],
        logits: Tensor,
        token_masks: Tensor,
        label_loss: float
    ):
        assert soft_gts.shape == logits.shape

        # 一般是置信度mask
        if soft_label_masks is not None:
            bp_masks = token_masks & soft_label_masks.flatten()
        flat_logits = logits.flatten(0, 1)[bp_masks]
        flat_soft_gts = soft_gts.flatten(0, 1)[bp_masks]
        if flat_soft_gts.size(0) != 0:
            probs = torch.log_softmax(flat_logits, dim=-1)
            pseudo_loss = F.kl_div(probs, flat_soft_gts, reduction="batchmean")

            alpha = self.criterion_cfg["loss_norm_alpha"]
            ema_decay = self.criterion_cfg["loss_ema_decay"]
            if self.criterion_cfg.get("loss_reweighting", True):
                # 注意这里要保证KL散度的计算没有问题，不能出现负的KL散度损失，不然系数会出现负数，导致优化方向相反
                assert pseudo_loss >= 0, "Got invalid loss = {}. Invalid gts: {}".format(pseudo_loss, flat_soft_gts)
                def ema_update(v: float, new_v: float, m: float):
                    return v * m + new_v * (1 - m)

                if sym_tbl().contains_global("label_loss"):
                    label_loss = ema_update(
                        sym_tbl().get_global("label_loss"), label_loss, ema_decay
                    )
                sym_tbl().set_global("label_loss", label_loss)
                pseudo_loss_item = pseudo_loss.item()
                if sym_tbl().contains_global("pseudo_loss"):
                    pseudo_loss_item = ema_update(
                        sym_tbl().get_global("pseudo_loss"), pseudo_loss_item, ema_decay
                    )
                sym_tbl().set_global("pseudo_loss", pseudo_loss_item)
                return pseudo_loss, 1.0 / (1 + alpha), alpha / (1.0 + alpha) * label_loss / pseudo_loss_item
            else:
                return pseudo_loss, 1.0, alpha
        else:
            return 0.0, 1.0, 0.0

    def nearest_matching(self, logits: Tensor, gts: Tensor):
        logits = einops.rearrange(logits, "b (k p) -> b k p", p=self.num_proto_per_type)
        target_indexes = gts.clone()
        for k in range(self.tagging_scheme.num_tags):
            ty_logits = logits[gts == k, k,:]       # gt为k的表征和k的原型之间的logits
            if ty_logits.size(0) == 0:
                continue
            indexes = torch.argmax(ty_logits, dim=-1)
            target_indexes[gts == k] = indexes + self.num_proto_per_type * k
        return target_indexes

    def sinkhorn_matching(self, logits: Tensor, gts: Tensor):
        from ...util import distributed_sinkhorn
        logits = einops.rearrange(logits, "b (k p) -> b k p", p=self.num_proto_per_type)
        target_indexes = gts.clone()
        for k in range(self.tagging_scheme.num_tags):
            ty_logits = logits[gts == k, k,:]       # gt为k的表征和k的原型之间的logits
            cost = -(1 - ty_logits)                 # 距离的相反数
            n_samples = cost.size(0)
            if n_samples == 0:
                continue
            q, indexes = distributed_sinkhorn(cost, sinkhorn_iterations=100)
            # q似乎是index的gumbel softmax的结果，我们不需要
            target_indexes[gts == k] = indexes + self.num_proto_per_type * k
        return target_indexes

    def sinkhorn_pot_matching(self, logits: Tensor, gts: Tensor):
        from ot import sinkhorn
        logits = einops.rearrange(logits, "b (k p) -> b k p", p=self.num_proto_per_type)
        target_indexes = gts.clone()
        for k in range(self.tagging_scheme.num_tags):
            ty_logits = logits[gts == k, k,:]       # gt为k的表征和k的原型之间的logits
            cost = 1 - ty_logits               # ot.sinkhorn要的是cost矩阵，即距离矩阵而不是相似度
            n_samples, n_proto = cost.size()
            if n_samples == 0:
                continue
            sample_constraint = torch.ones(n_samples, dtype=torch.float, device=cost.device)
            proto_constraint = torch.ones(n_proto, dtype=torch.float, device=cost.device)
            assignment = sinkhorn(
                a=sample_constraint,
                b=proto_constraint,
                M=cost / cost.max(),
                warn=False,
            )
            indexes = torch.argmax(assignment, dim=-1)
            target_indexes[gts == k] = indexes + self.num_proto_per_type * k
        return target_indexes

    def compact_loss(self, logits: Tensor, match_index: Tensor):
        matched_logits = []
        # NOTE: 有矩阵方法吗?
        for logits_, index in zip(logits, match_index):
            matched_logits.append(logits_[index])
        if len(matched_logits) == 0:
            loss = 0
        else:
            matched_logits = torch.stack(matched_logits)
            # 希望相似度增加
            loss = torch.pow(1 - matched_logits, 2)
            loss = torch.mean(loss)
        return loss, self.criterion_cfg["compact_weight"]

    def proto_cl_loss(self, ):
        proto_cl_type = self.criterion_cfg.get("proto_cl_type")
        if proto_cl_type is None:
            return 0, 0
        if proto_cl_type == "type":
            # 同类型不对比，只对比不同类型
            group_mask = torch.eye(
                self.tagging_scheme.num_tags,
                dtype=torch.bool, device=sym_tbl().device
            ).unsqueeze(-1).expand(
                -1, -1, self.num_proto_per_type
            ).flatten(1, 2).unsqueeze(1).expand(
                -1, self.num_proto_per_type, -1
            ).flatten(0, 1)
            """
            tensor([[ True,  True, False, False, False, False],
                    [ True,  True, False, False, False, False],
                    [False, False,  True,  True, False, False],
                    [False, False,  True,  True, False, False],
                    [False, False, False, False,  True,  True],
                    [False, False, False, False,  True,  True]])
            """
        elif proto_cl_type == "self":
            # 所有原型互相远离
            group_mask = torch.eye(
                self.tagging_scheme.num_tags * self.num_proto_per_type,
                dtype=torch.bool, device=sym_tbl().device
            )
        else:
            raise NotImplementedError("proto_cl_type {} not implemented".format(proto_cl_type))
        # 需要把原型互相拉远
        proto_sim_mat = self.model.similarity(self.model.type_ebd, self.model.type_ebd)
        # discard the main diagonal from both: labels and similarities matrix
        proto_sim_mat = proto_sim_mat[~group_mask].view(proto_sim_mat.shape[0], -1)       # (C * num_proto_per_type, (C - 1) * num_proto_per_type)
        # 不同类的相似度下降
        proto_cl_margin = self.criterion_cfg.get("proto_cl_margin")
        if proto_cl_margin is not None:
            proto_sim_mat = torch.max(proto_sim_mat, torch.as_tensor(proto_cl_margin))      # 相似度不需要无限小
        loss = torch.pow(1 + proto_sim_mat, 2)
        return torch.mean(loss), self.criterion_cfg["proto_cl_weight"]


@AlchemyModel.register("NonParamProtoTagger")
class AlchemyNonParamProtoTagger(AlchemyCosineProtoTagger):

    def __init__(self):
        AlchemyModel.__init__(self)
        # 跳过基类的初始化，这里不初始化原型
        task: NerTask = sym_tbl().task
        # scheme和模型绑定而不是和task绑定，所以应该放在模型里
        self.tagging_scheme = IOTaggingScheme(task.entity_types)     # 原型应该用IO tagging比较合适

        plm_type = self.model_cfg["plm_type"]
        if plm_type == "roberta":
            plm_cfg_cls = RobertaConfig
        elif plm_type == "bert":
            plm_cfg_cls = BertConfig
        else:
            raise ValueError("Unimplemented plm_type \"{}\"".format(plm_type))

        if "model_path" in self.model_cfg:
            self.config = plm_cfg_cls.from_pretrained(self.model_cfg["model_path"])
        else:
            # create model
            self.config = plm_cfg_cls.from_pretrained(self.model_cfg["plm_path"])

        # IO tagging，有多少tag就是多少type
        type_ebd = torch.randn(
            (self.tagging_scheme.num_tags * self.num_proto_per_type, self.config.hidden_size),
            requires_grad=False
        )

        if "model_path" in self.model_cfg:
            self.model = self.MODEL_CLASS.from_pretrained(
                self.model_cfg["model_path"],
                config=self.config,
                type_ebd=type_ebd,
                use_learnable_scalar=self.model_cfg["use_learnable_scalar"],

                # 一些可选的config，即如果文件中没有配置，那么就默认
                **filter_optional_cfg(self.model_cfg, {
                    "dropout",
                })
            )
        else:
            # create model
            self.model = self.MODEL_CLASS(
                config=self.config,
                type_ebd=type_ebd,
                use_learnable_scalar=self.model_cfg["use_learnable_scalar"],

                # 一些可选的config，即如果文件中没有配置，那么就默认
                **filter_optional_cfg(self.model_cfg, {
                    "dropout",
                })
            )

            logger.info("Use \"{}\" to initialize {} encoder".format(
                self.model_cfg["plm_path"], self.module.encoder.__class__.__name__
            ))
            self.module.encoder = self.module.encoder.__class__.from_pretrained(self.model_cfg["plm_path"])

    def forward(
        self, batch: MutableMapping[str, Any], needs_loss: bool, requires_grad: bool, **kwargs
    ) -> MutableMapping[str, Any]:
        batch = batch_to_device(batch, sym_tbl().device)

        hidden, unscaled_logits, logits = self.model.forward(
            encodings=batch["encoding"],
            encoding_masks=batch["encoding_masks"],
            token2start=batch["token2start"],
        )
        pred_tags = torch.div(torch.argmax(logits, dim=-1), self.num_proto_per_type, rounding_mode="floor")
        token_masks = batch['token_masks']

        outputs = {
            "hidden": hidden,
            "unscaled_logits": unscaled_logits,
            "logits": logits,
            "tags": pred_tags.detach().cpu().numpy(),
        }

        if needs_loss:
            losses = []
            loss_weights = []

            bp_masks = token_masks.flatten()     # bsz * n_tokens
            gts = batch["gt_seq_labels"]
            flat_logits = logits.flatten(0, 1)[bp_masks]
            flat_pred_tags = pred_tags.flatten()[bp_masks]
            flat_hiddens = hidden.flatten(0, 1)[bp_masks]
            flat_gts = gts.flatten()[bp_masks]

            # 负采样
            neg_ratio = self.criterion_cfg.get("neg_ratio", -1)
            if neg_ratio < 0:
                pass
            else:
                # 做一些随机负采样
                sample_masks = (flat_gts != 0) | (torch.rand_like(flat_gts, dtype=torch.float) < neg_ratio)
                flat_logits = torch.masked_select(flat_logits, sample_masks.unsqueeze(-1)).view(-1, logits.size(-1))
                flat_pred_tags = torch.masked_select(flat_pred_tags, sample_masks)
                flat_hiddens = torch.masked_select(flat_hiddens, sample_masks.unsqueeze(-1)).view(-1, flat_hiddens.size(-1))
                flat_gts = torch.masked_select(flat_gts, sample_masks)

            match_index = self.sinkhorn_matching_and_ema(flat_hiddens, flat_logits, flat_pred_tags, flat_gts)
            if self.criterion_cfg["type"] == "ce":
                loss_fct = CrossEntropyLoss()
            elif self.criterion_cfg["type"] == "gce":
                gce_q = self.criterion_cfg.get("q")
                raise NotImplementedError()
            else:
                raise NotImplementedError(f"Unknown criterion {self.criterion_cfg}")
            label_loss = loss_fct.forward(flat_logits, match_index)
            losses.append(label_loss)
            loss_weights.append(1.0)

            if "soft_gt_seq_labels" in batch:
                pseudo_loss, label_weight, pseudo_weight = self.pseudo_loss(
                    soft_gts=batch["soft_gt_seq_labels"],
                    soft_label_masks=batch.get("label_masks"),       # 一般是置信度mask
                    logits=logits,
                    token_masks=bp_masks,
                    label_loss=label_loss.item(),
                )
                loss_weights[-1] = label_weight
                losses.append(pseudo_loss)
                loss_weights.append(pseudo_weight)

            compact_loss, compact_weight = self.compact_loss(flat_logits, match_index)
            losses.append(compact_loss)
            loss_weights.append(compact_weight)

            outputs["loss"] = self.backward.backward(losses, loss_weights, requires_grad=requires_grad)
        return outputs

    def sinkhorn_matching_and_ema(self, hiddens: Tensor, logits: Tensor, pred_tags: Tensor, gts: Tensor):
        from ...util import distributed_sinkhorn
        momentum = self.criterion_cfg["proto_ema_momentum"]

        def ema_update(v: float, new_v: float):
            return v * momentum + new_v * (1 - momentum)

        well_pred_mask = pred_tags == gts       # 预测对的mask，只有预测对的hidden state才能更新原型
        logits = einops.rearrange(logits, "b (k p) -> b k p", p=self.num_proto_per_type)
        target_indexes = gts.clone()
        protos = self.model.type_ebd.clone()
        protos = einops.rearrange(protos, "(k p) d -> k p d", p=self.num_proto_per_type)
        for k in range(self.tagging_scheme.num_tags):
            mask_k = gts == k
            ty_logits = logits[mask_k, k,:]         # gt为k的表征和k的原型之间的logits
            cost = -(1 - ty_logits)                 # 距离的相反数
            n_samples = cost.size(0)
            if n_samples == 0:
                continue
            q, indexes = distributed_sinkhorn(cost, sinkhorn_iterations=100)
            # 依据匹配选定target
            target_indexes[mask_k] = indexes + self.num_proto_per_type * k
            # 依据匹配更新原型
            well_pred_mask_k = well_pred_mask[mask_k]       # gt为k的token，是否预测正确
            hiddens_k = hiddens[mask_k]                     # gt为k的token的表征
            well_pred_proto_mask_k = q * einops.repeat(
                well_pred_mask_k, "n -> n p", p=self.num_proto_per_type
            )       # 预测正确且为k的token的原型匹配矩阵
            well_pred_hiddens_k = hiddens_k * well_pred_mask_k.unsqueeze(-1)
            # 每个原型匹配到的表征的sum
            matched_hidden_sum = torch.einsum("np,nd->pd", well_pred_proto_mask_k, well_pred_hiddens_k)
            match_cnt = torch.sum(well_pred_proto_mask_k, dim=0)        # 每个原型匹配到多少个
            if torch.sum(match_cnt) > 0:
                update_mask = match_cnt != 0
                # normalize可以理解为一种mean
                matched_hidden_mean = F.normalize(matched_hidden_sum, p=2, dim=-1)
                newv = ema_update(protos[k, update_mask], matched_hidden_mean[update_mask])
                protos[k, update_mask] = newv
        # NOTE: 注意这里不能ddp，因为proto的更新没有走accelerate的流程，ddp需要特殊处理
        protos = einops.rearrange(protos, "k p d -> (k p) d")
        self.model.type_ebd = torch.nn.Parameter(F.normalize(protos, dim=-1), requires_grad=False)

        return target_indexes


@AlchemyModel.register("L2NonParamProtoTagger")
class AlchemyL2NonParamProtoTagger(AlchemyNonParamProtoTagger):
    MODEL_CLASS = L2ProtoTagger

    def sinkhorn_matching_and_ema(self, hiddens: Tensor, logits: Tensor, pred_tags: Tensor, gts: Tensor):
        from ...util import distributed_sinkhorn
        momentum = self.criterion_cfg["proto_ema_momentum"]

        def ema_update(v: float, new_v: float):
            return v * momentum + new_v * (1 - momentum)

        well_pred_mask = pred_tags == gts       # 预测对的mask，只有预测对的hidden state才能更新原型
        logits = einops.rearrange(logits, "b (k p) -> b k p", p=self.num_proto_per_type)
        assert torch.all(logits < 0)            # 假设前面用的是-cdist，后面的regularization需要都是负的的假设
        logits = logits / -torch.min(logits)    # 归一化，压缩到[-1,0]之间，防止sinkhorn的exp产生数值错误
        target_indexes = gts.clone()
        protos = self.model.type_ebd.clone()
        protos = einops.rearrange(protos, "(k p) d -> k p d", p=self.num_proto_per_type)
        for k in range(self.tagging_scheme.num_tags):
            mask_k = gts == k
            ty_logits = logits[mask_k, k,:]         # gt为k的表征和k的原型之间的logits
            cost = ty_logits                        # logits是相似度，distributed_sinkhorn的代价和pot sinkhorn的代价是相反的
            n_samples = cost.size(0)
            if n_samples == 0:
                continue
            q, indexes = distributed_sinkhorn(cost, sinkhorn_iterations=100)
            # 依据匹配选定target
            target_indexes[mask_k] = indexes + self.num_proto_per_type * k
            # 依据匹配更新原型
            well_pred_mask_k = well_pred_mask[mask_k]       # gt为k的token，是否预测正确
            hiddens_k = hiddens[mask_k]                     # gt为k的token的表征
            well_pred_proto_mask_k = q * einops.repeat(
                well_pred_mask_k, "n -> n p", p=self.num_proto_per_type
            )       # 预测正确且为k的token的原型匹配矩阵
            well_pred_hiddens_k = hiddens_k * well_pred_mask_k.unsqueeze(-1)
            # 每个原型匹配到的表征的sum
            matched_hidden_sum = torch.einsum("np,nd->pd", well_pred_proto_mask_k, well_pred_hiddens_k)
            match_cnt = torch.sum(well_pred_proto_mask_k, dim=0)        # 每个原型匹配到多少个
            matched_hidden_mean = matched_hidden_sum / match_cnt.unsqueeze(-1)
            if torch.sum(match_cnt) > 0:
                update_mask = match_cnt != 0
                newv = ema_update(protos[k, update_mask], matched_hidden_mean[update_mask])
                protos[k, update_mask] = newv
        # NOTE: 注意这里不能ddp，因为proto的更新没有走accelerate的流程，ddp需要特殊处理
        protos = einops.rearrange(protos, "k p d -> (k p) d")
        self.model.type_ebd = torch.nn.Parameter(protos, requires_grad=False)

        return target_indexes

    def compact_loss(self, logits: Tensor, match_index: Tensor):
        matched_logits = []
        # NOTE: 有矩阵方法吗?
        for logits_, index in zip(logits, match_index):
            matched_logits.append(logits_[index])
        if len(matched_logits) == 0:
            loss = 0
        else:
            matched_logits = torch.stack(matched_logits)
            # 希望相似度增加
            loss = -torch.mean(matched_logits)
        return loss, self.criterion_cfg["compact_weight"]
