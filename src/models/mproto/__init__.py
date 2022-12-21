from pathlib import Path
from typing import Any, Dict, MutableMapping, Optional
from loguru import logger

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss, KLDivLoss, Embedding
import torch.nn.functional as F
from transformers import BertConfig, RobertaConfig
import numpy as np

from alchemy import AlchemyModel, sym_tbl
from alchemy.pipeline import OutputPipeline
from alchemy.util import batch_to_device, filter_optional_cfg
from alchemy.util.optim import prepare_trf_based_model_params
from ...task.ner.tagging import BioTaggingScheme, IOTaggingScheme, TaggingScheme
from ...task.ner import NerTask
from .model import CosineProtoTagger, CosineProtoTaggerWithPseudo
import einops
from ot import sinkhorn
from ...util.sinkhorn import SinkhornDistance



@OutputPipeline.register()
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


@AlchemyModel.register("BaseMProtoTagger")
class AlchemyBaseMProtoTagger(AlchemyModel):
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

        # IO tagging，有多少tag就是多少type
        protos = torch.randn(
            (self.tagging_scheme.num_tags * self.num_proto_per_type, self.config.hidden_size),
            requires_grad=False
        )

        if "model_path" in self.model_cfg:
            self.model = self.MODEL_CLASS.from_pretrained(
                self.model_cfg["model_path"],
                config=self.config,
                protos=protos,
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
                protos=protos,
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

        hidden, _, logits = self.model.forward(
            encodings=batch["encoding"],
            encoding_masks=batch["encoding_masks"],
            token2start=batch["token2start"],
        )
        token_masks = batch['token_masks']
        pred_tags = torch.div(torch.argmax(logits, dim=-1), self.num_proto_per_type, rounding_mode="floor")

        outputs = {
            "hidden": hidden,
            # "unscaled_logits": unscaled_logits,
            "logits": logits,
            "tags": pred_tags.detach().cpu().numpy(),
        }

        if needs_loss:
            losses = []
            loss_weights = []

            flat_token_masks = token_masks.flatten()     # bsz * n_tokens
            gts = batch["gt_seq_labels"]
            flat_logits = logits.flatten(0, 1)[flat_token_masks]
            flat_pred_tags = pred_tags.flatten()[flat_token_masks]
            flat_hiddens = hidden.flatten(0, 1)[flat_token_masks]
            flat_gts = gts.flatten()[flat_token_masks]

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

            match_index = self.sinkhorn_matching_and_ema(
                flat_hiddens,
                flat_logits,
                flat_pred_tags,
                flat_gts
            )

            if self.criterion_cfg["type"] == "ce":
                loss_fct = CrossEntropyLoss()
            else:
                raise NotImplementedError(f"Unknown criterion {self.criterion_cfg}")
            label_loss = loss_fct.forward(flat_logits, match_index)
            losses.append(label_loss)
            loss_weights.append(1.0)

            compact_loss, compact_weight = self.compact_loss(flat_logits, match_index)
            losses.append(compact_loss)
            loss_weights.append(compact_weight)

            outputs["loss"] = self.backward.backward(losses, loss_weights, requires_grad=requires_grad)
        return outputs

    def sinkhorn_matching_and_ema(
        self,
        hiddens: Tensor,
        logits: Tensor,
        pred_tags: Tensor,
        gts: Tensor
    ):
        def ema_update(v: float, new_v: float):
            momentum = self.criterion_cfg["proto_ema_momentum"]
            return v * momentum + new_v * (1 - momentum)

        well_pred_mask = pred_tags == gts       # 预测对的mask，只有预测对的hidden state才能更新原型
        logits = einops.rearrange(logits, "b (k p) -> b k p", p=self.num_proto_per_type)
        target_indexes = gts.clone()
        protos = self.model.protos.clone()
        protos = einops.rearrange(protos, "(k p) d -> k p d", p=self.num_proto_per_type)
        for k in range(self.tagging_scheme.num_tags):
            mask_k = gts == k
            ty_logits = logits[mask_k, k,:]         # gt为k的表征和k的原型之间的logits
            cost = (1 - ty_logits)                  # 距离
            n_samples, n_protos = cost.size()
            if n_samples == 0:
                continue

            ratios = [1 / self.num_proto_per_type for _ in range(self.num_proto_per_type)]
            proto_constraint = torch.as_tensor(ratios, dtype=torch.float, device=cost.device) * n_samples
            sample_constraint = torch.ones(n_samples, dtype=torch.float, device=cost.device)
            sinkhorn_type = self.criterion_cfg.get("sinkhorn_type", "solver")
            if sinkhorn_type == "solver":
                _, assignment = SinkhornDistance().forward(
                    sample_constraint, proto_constraint, cost
                )
            elif sinkhorn_type == "pot":
                assignment = sinkhorn(
                    a=sample_constraint,
                    b=proto_constraint,
                    M=cost / abs(cost.max()),
                    reg=0.001,
                    verbose=False,
                    log=False,
                    warn=False,
                )
            else:
                raise ValueError("Unknown sinkhorn type {}".format(sinkhorn_type))
            indexes = torch.argmax(assignment, dim=-1)
            onehot_indexes = F.one_hot(indexes, assignment.size(-1)).float()
            # onehot_indexes = F.gumbel_softmax(assignment, tau=0.5, hard=True)

            # 依据匹配选定target
            well_pred_mask_k = well_pred_mask[mask_k]       # gt为k的token，是否预测正确

            well_pred_proto_mask_k = onehot_indexes * einops.repeat(
                well_pred_mask_k, "n -> n p", p=self.num_proto_per_type
            )       # 预测正确且为k的token的原型匹配矩阵
            hiddens_k = hiddens[mask_k]                     # gt为k的token的表征
            well_pred_hiddens_k = hiddens_k * well_pred_mask_k.unsqueeze(-1)
            # 每个原型匹配到的表征的sum
            matched_hidden_sum = torch.einsum("np,nd->pd", well_pred_proto_mask_k, well_pred_hiddens_k)
            match_cnt = torch.sum(well_pred_proto_mask_k, dim=0)        # 每个原型匹配到多少个

            # 将target设置为匹配的结果
            target_indexes[mask_k] = indexes + self.num_proto_per_type * k

            # 更新原型
            if torch.sum(match_cnt) > 0:
                update_mask = match_cnt != 0
                # normalize可以理解为一种mean
                matched_hidden_mean = F.normalize(matched_hidden_sum, p=2, dim=-1)
                newv = ema_update(protos[k, update_mask], matched_hidden_mean[update_mask])
                protos[k, update_mask] = newv
        # NOTE: 注意这里不能ddp，因为proto的更新没有走accelerate的流程，ddp需要特殊处理
        protos = einops.rearrange(protos, "k p d -> (k p) d")
        self.model.protos = torch.nn.Parameter(F.normalize(protos, dim=-1), requires_grad=False)

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


@AlchemyModel.register("MProtoTagger")
class AlchemyMProtoTagger(AlchemyBaseMProtoTagger):
    def forward(
        self, batch: MutableMapping[str, Any], needs_loss: bool, requires_grad: bool, **kwargs
    ) -> MutableMapping[str, Any]:
        batch = batch_to_device(batch, sym_tbl().device)

        hidden, _, logits = self.model.forward(
            encodings=batch["encoding"],
            encoding_masks=batch["encoding_masks"],
            token2start=batch["token2start"],
        )
        token_masks = batch['token_masks']
        pred_tags = torch.div(torch.argmax(logits, dim=-1), self.num_proto_per_type, rounding_mode="floor")

        outputs = {
            "hidden": hidden,
            # "unscaled_logits": unscaled_logits,
            "logits": logits,
            "tags": pred_tags.detach().cpu().numpy(),
        }

        if needs_loss:
            losses = []
            loss_weights = []

            flat_token_masks = token_masks.flatten()     # bsz * n_tokens
            gts = batch["gt_seq_labels"]
            flat_logits = logits.flatten(0, 1)[flat_token_masks]
            flat_pred_tags = pred_tags.flatten()[flat_token_masks]
            flat_hiddens = hidden.flatten(0, 1)[flat_token_masks]
            flat_gts = gts.flatten()[flat_token_masks]

            match_index, bp_masks = self.sinkhorn_matching_and_ema(
                flat_hiddens,
                flat_logits,
                flat_pred_tags,
                flat_gts
            )

            flat_logits = torch.masked_select(flat_logits, bp_masks.unsqueeze(-1)).view(-1, logits.size(-1))
            match_index = torch.masked_select(match_index, bp_masks)

            if self.criterion_cfg["type"] == "ce":
                loss_fct = CrossEntropyLoss()
            else:
                raise NotImplementedError(f"Unknown criterion {self.criterion_cfg}")
            label_loss = loss_fct.forward(flat_logits, match_index)
            losses.append(label_loss)
            loss_weights.append(1.0)

            compact_loss, compact_weight = self.compact_loss(flat_logits, match_index)
            losses.append(compact_loss)
            loss_weights.append(compact_weight)

            outputs["loss"] = self.backward.backward(losses, loss_weights, requires_grad=requires_grad)
        return outputs

    def sinkhorn_matching_and_ema(
        self,
        hiddens: Tensor,
        logits: Tensor,
        pred_tags: Tensor,
        gts: Tensor
    ):
        def ema_update(v: float, new_v: float):
            momentum = self.criterion_cfg["proto_ema_momentum"]
            return v * momentum + new_v * (1 - momentum)

        bp_masks = torch.ones_like(gts, dtype=torch.bool)
        well_pred_mask = pred_tags == gts       # 预测对的mask，只有预测对的hidden state才能更新原型
        logits = einops.rearrange(logits, "b (k p) -> b k p", p=self.num_proto_per_type)
        target_indexes = gts.clone()
        protos = self.model.protos.clone()
        protos = einops.rearrange(protos, "(k p) d -> k p d", p=self.num_proto_per_type)
        for k in range(self.tagging_scheme.num_tags):
            mask_k = gts == k
            if k == 0:
                # 其他类别的原型作为pseudo，真实在前pseudo在后
                ty_logits = logits[mask_k].flatten(1, 2)
            else:
                ty_logits = logits[mask_k, k,:]         # gt为k的表征和k的原型之间的logits
            cost = (1 - ty_logits)                  # 距离
            n_samples, n_protos = cost.size()
            if n_samples == 0:
                continue

            if k == 0:
                # 设置伪原型的分配比例
                none_ratio = self.criterion_cfg.get("none_ratio", 0.0)
                assert 0 <= none_ratio <= 1, "Invalid none_ratio = {}".format(none_ratio)
                ratios = [
                    none_ratio / self.num_proto_per_type for _ in range(self.num_proto_per_type)
                ] + [
                    (1 - none_ratio) / (n_protos - self.num_proto_per_type) for _ in range(n_protos - self.num_proto_per_type)
                ]
            else:
                ratios = [1 / self.num_proto_per_type for _ in range(self.num_proto_per_type)]
            proto_constraint = torch.as_tensor(ratios, dtype=torch.float, device=cost.device) * n_samples
            sample_constraint = torch.ones(n_samples, dtype=torch.float, device=cost.device)
            sinkhorn_type = self.criterion_cfg.get("sinkhorn_type", "solver")
            if sinkhorn_type == "solver":
                _, assignment = SinkhornDistance().forward(
                    sample_constraint, proto_constraint, cost
                )
            elif sinkhorn_type == "pot":
                assignment = sinkhorn(
                    a=sample_constraint,
                    b=proto_constraint,
                    M=cost / abs(cost.max()),
                    reg=0.001,
                    verbose=False,
                    log=False,
                    warn=False,
                )
            else:
                raise ValueError("Unknown sinkhorn type {}".format(sinkhorn_type))
            indexes = torch.argmax(assignment, dim=-1)
            onehot_indexes = F.one_hot(indexes, assignment.size(-1)).float()
            # onehot_indexes = F.gumbel_softmax(assignment, tau=0.5, hard=True)

            if k == 0:
                # 将匹配到伪原型的特征剔除出去，并且添加bp_mask
                # 依据匹配选定target
                match_true_none_mask = indexes < self.num_proto_per_type         # 真的匹配到none，而不是pseudo
                logger.debug("Neg rate = {}".format(torch.sum(match_true_none_mask) / indexes.size(0)))
                bp_masks[mask_k] = match_true_none_mask     # 匹配到伪原型的不参与bp

                well_pred_mask_k = well_pred_mask[mask_k] & match_true_none_mask       # gt为none，预测为none，且匹配到true none原型
                onehot_indexes = onehot_indexes[..., :self.num_proto_per_type]          # 不考虑伪原型的匹配，伪原型不通过匹配来更新
            else:
                # 依据匹配选定target
                well_pred_mask_k = well_pred_mask[mask_k]       # gt为k的token，是否预测正确

            well_pred_proto_mask_k = onehot_indexes * einops.repeat(
                well_pred_mask_k, "n -> n p", p=self.num_proto_per_type
            )       # 预测正确且为k的token的原型匹配矩阵
            hiddens_k = hiddens[mask_k]                     # gt为k的token的表征
            well_pred_hiddens_k = hiddens_k * well_pred_mask_k.unsqueeze(-1)
            # 每个原型匹配到的表征的sum
            matched_hidden_sum = torch.einsum("np,nd->pd", well_pred_proto_mask_k, well_pred_hiddens_k)
            match_cnt = torch.sum(well_pred_proto_mask_k, dim=0)        # 每个原型匹配到多少个

            # 将target设置为匹配的结果
            target_indexes[mask_k] = indexes + self.num_proto_per_type * k

            # 更新原型
            if torch.sum(match_cnt) > 0:
                update_mask = match_cnt != 0
                # normalize可以理解为一种mean
                matched_hidden_mean = F.normalize(matched_hidden_sum, p=2, dim=-1)
                newv = ema_update(protos[k, update_mask], matched_hidden_mean[update_mask])
                protos[k, update_mask] = newv
        # NOTE: 注意这里不能ddp，因为proto的更新没有走accelerate的流程，ddp需要特殊处理
        protos = einops.rearrange(protos, "k p d -> (k p) d")
        self.model.protos = torch.nn.Parameter(F.normalize(protos, dim=-1), requires_grad=False)

        return target_indexes, bp_masks
