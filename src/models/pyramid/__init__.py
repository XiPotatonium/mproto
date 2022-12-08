from typing import Any, Dict, List, MutableMapping
from loguru import logger

import torch
from torch import Tensor
from torch.nn import Embedding
import torch.nn.functional as F
from transformers import BertConfig, RobertaConfig

from alchemy import AlchemyModel, sym_tbl
from alchemy.util import batch_to_device, filter_optional_cfg
from alchemy.util.optim import prepare_trf_based_model_params
from alchemy.pipeline import OutputPipeline
from ...task.ner import NerTask
from ...criterion.gce_loss import generalized_cross_entropy
from .model import BiPyramid


@OutputPipeline.register("ProcBiPyramid")
class ProcBiPyramid(OutputPipeline):
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, outputs: Any, inputs: MutableMapping[str, Any]) -> Any:
        pred_logits = outputs["logits_list"]  # list of each pyramid layer
        pred_masks = outputs["mask_list"]  # padding masks

        ret: List[List[Dict[str, Any]]] = []
        for layer_i, (layer_logits, layer_padding_masks) in enumerate(zip(pred_logits, pred_masks)):
            for sample_i, (sample_layer_logits, sample_layer_padding_masks) in enumerate(
                zip(layer_logits.detach().cpu(), layer_padding_masks)
            ):
                if layer_i == 0:
                    ret.append([])
                layer_scores = torch.softmax(sample_layer_logits, dim=-1)
                layer_score, layer_type = torch.max(layer_scores, dim=-1)
                for left, (scores, score, type_index, padding_mask) in enumerate(
                    zip(layer_scores, layer_score, layer_type, sample_layer_padding_masks)
                ):
                    if padding_mask:
                        continue
                    ret[sample_i].append({
                        "start": left,
                        "end": left + layer_i,
                        "type": type_index.item(),
                        "type_score": score.item(),
                        "type_scores": scores.numpy(),
                    })

        return ret


@AlchemyModel.register("BiPyramid")
class AlchemyBiPyramid(AlchemyModel):

    def __init__(self):
        super().__init__()
        task: NerTask = sym_tbl().task

        token_ebd = None
        pos_ebd = None
        char_ebd = None
        if self.model_cfg["use_w2v"]:
            if "w2v_size" in self.model_cfg:
                logger.warning("Initialize ebd with size {}".format(self.model_cfg["w2v_size"]))
                token_ebd = Embedding(*self.model_cfg["w2v_size"])
            elif task.token_ebd is not None:
                token_ebd = Embedding.from_pretrained(torch.from_numpy(task.token_ebd).float(), freeze=False)
            else:
                logger.warning("Using pseudo token embedding")
                token_ebd = Embedding(0, 100)
        if self.model_cfg["use_pos"]:
            if "pos_size" in self.model_cfg:
                logger.warning("Initialize ebd with size {}".format(self.model_cfg["pos_size"]))
                pos_ebd = Embedding(*self.model_cfg["pos_size"])
            elif "pos_dim" in self.model_cfg:
                pos_ebd = Embedding(len(task.pos_vocab), self.model_cfg["pos_dim"])
            else:
                logger.warning("Using pseudo pos embedding")
                pos_ebd = Embedding(0, 100)
        if self.model_cfg["use_char"]:
            if "char_size" in self.model_cfg:
                logger.warning("Initialize ebd with size {}".format(self.model_cfg["char_size"]))
                char_ebd = Embedding(*self.model_cfg["char_size"])
            elif "char_dim" in self.model_cfg:
                char_ebd = Embedding(len(task.char_vocab), self.model_cfg["char_dim"])
            else:
                logger.warning("Using pseudo char embedding")
                char_ebd = Embedding(0, 100)

        plm_type = self.model_cfg["plm_type"]
        if plm_type == "roberta":
            plm_cfg_cls = RobertaConfig
        elif plm_type == "bert":
            plm_cfg_cls = BertConfig
        else:
            raise ValueError("Unimplemented plm_type \"{}\"".format(plm_type))

        if self.model_cfg.get("model_path", None):
            self.config = plm_cfg_cls.from_pretrained(self.model_cfg["model_path"])
            self.model = BiPyramid.from_pretrained(
                self.model_cfg["model_path"],
                ignore_mismatched_sizes=self.model_cfg.get("ignore_mismatched_sizes", False),

                config=self.config,
                num_types=task.num_entity_types,
                max_depth=self.model_cfg["max_depth"],
                use_lstm=self.model_cfg["use_lstm"],
                token_ebd=token_ebd,
                pos_ebd=pos_ebd,
                char_ebd=char_ebd,

                # 一些可选的config，即如果文件中没有配置，那么就默认
                **filter_optional_cfg(self.model_cfg, {
                    "dropout", "pool_type",
                    "lstm_layers", "lstm_drop",
                    "char_lstm_layers", "char_lstm_drop",
                })
            )
        else:
            # create model
            self.config = plm_cfg_cls.from_pretrained(self.model_cfg["plm_path"])
            self.model = BiPyramid(
                config=self.config,
                num_types=task.num_entity_types,
                max_depth=self.model_cfg["max_depth"],
                use_lstm=self.model_cfg["use_lstm"],
                token_ebd=token_ebd,
                pos_ebd=pos_ebd,
                char_ebd=char_ebd,

                # 一些可选的config，即如果文件中没有配置，那么就默认
                **filter_optional_cfg(self.model_cfg, {
                    "dropout", "pool_type",
                    "lstm_layers", "lstm_drop",
                    "char_lstm_layers", "char_lstm_drop",
                })
            )

            self.model.encoder.with_pretrained(path=self.model_cfg["plm_path"])

    def max_positions(self):
        return self.config.max_position_embeddings

    def optim_params(self, **kwargs):
        return prepare_trf_based_model_params(
            self.model,
            self.module.encoder.trf,
            **filter_optional_cfg(kwargs, {"weight_decay", "trf_lr"}),
        )

    def set_requires_grad(self, requires_grad: bool, mode: str, **kwargs):
        if mode == "encoder.trf":
            for _, param in self.module.encoder.trf.named_parameters():
                param.requires_grad = requires_grad
        elif mode == "all":
            for _, param in self.module.named_parameters():
                param.requires_grad = requires_grad
        else:
            raise NotImplementedError(mode)

    def forward(
        self,
        batch: MutableMapping[str, Any],
        needs_loss: bool,
        requires_grad: bool,
        **kwargs
    ) -> MutableMapping[str, Any]:
        batch = batch_to_device(batch, sym_tbl().device)
        bsz, _ = batch["encoding"].size()
        logits_list, mask_list = self.model.forward(
            encodings=batch["encoding"],
            encoding_masks=batch["encoding_masks"],
            token2encoding_masks=batch["token2encoding_masks"],
            token2start=batch["token2start"],
            token_masks=batch['token_masks'],
            pos_encoding=batch['pos_encoding'],
            w2v_encoding=batch["w2v_encoding"],
        )
        outputs = {
            "logits_list": logits_list,
            "mask_list": mask_list,
        }

        if needs_loss:
            loss = self._compute_loss(batch, logits_list, mask_list)
            outputs["loss"] = self.backward.backward(loss, requires_grad=requires_grad)
        return outputs

    def _compute_loss(self, batch, logits: List[Tensor], padding_masks: List[Tensor]):
        # pred_logits: [(bsz, lay_sent_len, n_cls)...n_fpn_layers]

        n_fpn_layers = len(logits)
        gt_types = batch["gt_types"]
        gts = [
            torch.zeros((l_logits.size()[:2]), dtype=gt_types.dtype, device=l_logits.device)
            for l_logits in logits
        ]
        # NOTE: 注意这里的gt中是可能包含负样本的，因此需要构造gt_masks而不是直接用positive_masks
        gt_masks = [
            torch.zeros((l_logits.size()[:2]), dtype=torch.bool, device=l_logits.device)
            for l_logits in logits
        ]

        for sample_i, (sample_gt_spans, sample_gt_types, sample_mask) in enumerate(
            zip(batch["gt_spans"], gt_types, batch["gt_masks"])
        ):
            sample_gt_spans = torch.masked_select(
                sample_gt_spans, sample_mask.unsqueeze(-1)
            ).view(-1, 2)
            sample_gt_types = torch.masked_select(sample_gt_types, sample_mask)
            for (left, right), ty in zip(sample_gt_spans, sample_gt_types):
                if right - left < n_fpn_layers:
                    gts[right - left][sample_i][left] = ty
                    gt_masks[right - left][sample_i][left] = 1
        gts = torch.cat(gts, dim=1)         # bsz, n_pred
        gt_masks = torch.cat(gt_masks, dim=1)         # bsz, n_pred
        logits = torch.cat(logits, dim=1)     # bsz, n_pred, n_cls

        flat_padding_masks = torch.cat(padding_masks, dim=1)     # bsz, n_pred
        flat_masks = ~flat_padding_masks

        num_pos_spans = torch.sum(gts != 0)
        neg_ratio = self.criterion_cfg.get("neg_ratio", -1)
        if neg_ratio < 0:
            pass
        elif neg_ratio == 0:
            flat_masks = flat_masks & gt_masks        # 只bp gt
        else:
            # 做一些随机负采样
            # 根据Empirical Analysis of Unlabeled Entity Problem in Named Entity Recognition
            # 我们最好采 \lambda * n, where n = token_count 个负样本
            # 为了方便，我们在所有span上做负采样，为了保证概率一致，即\lambda1 * n = \lambda2 * (n + max(n - l + 1, 1)) * min(l, n) / 2
            token_count = batch["token_masks"].float().sum(-1, keepdim=True)
            n_fpn_layers = torch.ones_like(token_count) * n_fpn_layers
            neg_ratio = token_count * neg_ratio * 2 / torch.minimum(token_count, n_fpn_layers) / (
                token_count + torch.maximum(token_count - n_fpn_layers + 1, torch.ones_like(token_count))
            )
            flat_masks = gt_masks | ((torch.rand_like(gts, dtype=torch.float) < neg_ratio) & flat_masks)

        flat_logits = torch.masked_select(
            logits, mask=flat_masks.unsqueeze(-1)
        ).view(-1, logits.size()[-1])
        flat_gts = torch.masked_select(gts, mask=flat_masks)

        weight = torch.ones(
            logits.size()[-1],
            dtype=logits.dtype,
            device=logits.device
        )
        nil_weight = self.criterion_cfg.get("nil_weight", -1)
        if nil_weight < 0:
            if flat_gts.size(0) > num_pos_spans:
                # NOTE: 这里做了一个修正，本来cond只有self.nil_weight < 0，
                # 但是在n_gts很大的时候，甚至大于等于n_candidates的时候，原来的公式会计算出inf或者负的weight，这是不合理的
                # 考虑到这个weight是在nil太多的时候（n_candidates << n_gts)平衡nil的权重，而nil比较小的时候，权重和其他（1.0）一样就好了
                weight[0] = num_pos_spans / (flat_gts.size(0) - num_pos_spans)
        else:
            weight[0] = nil_weight

        if self.criterion_cfg["type"] == "gce":
            q = self.criterion_cfg["q"]
            if nil_weight != 1.0:
                raise NotImplementedError("GCE loss only support nil_weight = 1.0")
            loss = generalized_cross_entropy(
                logits=flat_logits,
                targets=flat_gts,
                q=q,
                weight=weight,
                reduction="mean"
            )
        elif self.criterion_cfg["type"] == "ce":
            loss = F.cross_entropy(flat_logits, flat_gts, weight=weight, reduction="mean")
        else:
            raise ValueError(f"Unknown criterion {self.criterion_cfg}")
        return loss
