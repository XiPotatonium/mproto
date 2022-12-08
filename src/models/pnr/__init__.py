from typing import Any, Dict, List, Mapping, MutableMapping
from loguru import logger

import torch
from torch import Tensor
from torch.nn import Embedding
import torch.nn.functional as F
from transformers import BertConfig, RobertaConfig

from alchemy import AlchemyModel, sym_tbl
from alchemy.pipeline import OutputPipeline
from alchemy.util import batch_to_device, filter_optional_cfg
from alchemy.util.optim import prepare_trf_based_model_params
from ...task.ner import NerTask
from ...criterion.set_criterion import NerSetCriterion
from .model import PnRNet
from .util import generate_boundary


@OutputPipeline.register("ProcPnROutput")
class ProcPnROutput(OutputPipeline):
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, outputs: Any, inputs: MutableMapping[str, Any]) -> Any:
        ret = []

        last_layer_outputs = outputs["stage2"][-1]

        all_cls_scores = torch.softmax(last_layer_outputs["entity_logits"], dim=-1)
        all_left_scores = torch.softmax(last_layer_outputs["p_left"], dim=-1)
        all_right_scores = torch.softmax(last_layer_outputs["p_right"], dim=-1)
        all_cls_score, entity_type = torch.max(all_cls_scores, dim=-1)
        all_left_score, span_left = torch.max(all_left_scores, dim=-1)
        all_right_score, span_right = torch.max(all_right_scores, dim=-1)
        for sample_i, (
            s_cls_scores, s_left_scores, s_right_scores,
            s_cls_score, s_entity_type, s_left_score, s_left, s_right_score, s_right
        ) in enumerate(zip(
            all_cls_scores, all_left_scores, all_right_scores,
            all_cls_score, entity_type, all_left_score, span_left, all_right_score, span_right
        )):
            sample_outputs = []
            for q_i, (
                q_cls_scores, q_left_scores, q_right_scores,
                q_cls_score, q_entity_type, q_left_score, q_left, q_right_score, q_right
            ) in enumerate(zip(
                s_cls_scores, s_left_scores, s_right_scores,
                s_cls_score, s_entity_type, s_left_score, s_left, s_right_score, s_right
            )):
                sample_outputs.append({
                    "start": q_left.item(),
                    "end": q_right.item(),
                    "type": q_entity_type.item(),
                    "start_score": q_left_score.item(),
                    "end_score": q_right_score.item(),
                    "type_score": q_cls_score.item(),
                    "start_scores": q_left_scores.detach().cpu().numpy(),
                    "end_scroes": q_right_scores.detach().cpu().numpy(),
                    "type_scores": q_cls_scores.detach().cpu().numpy(),
                })
            ret.append(sample_outputs)
        return ret


@AlchemyModel.register("PnRNet")
class AlchemyPnRNet(AlchemyModel):
    MODEL_CLASS = PnRNet

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

        if "model_path" in self.model_cfg:
            logger.info("Load pretrained checkpoint from {}".format(self.model_cfg["model_path"]))
            self.config = plm_cfg_cls.from_pretrained(self.model_cfg["model_path"])
            self.model = self.MODEL_CLASS.from_pretrained(
                self.model_cfg["model_path"],
                ignore_mismatched_sizes=self.model_cfg.get("ignore_mismatched_sizes", False),

                config=self.config,
                n_classes=task.num_entity_types,
                n_queries=self.model_cfg["num_entity_queries"],
                use_lstm=self.model_cfg["use_lstm"],
                token_ebd=token_ebd,
                pos_ebd=pos_ebd,
                char_ebd=char_ebd,

                fpn_type=self.model_cfg["fpn_type"],
                fpn_layers=self.model_cfg["fpn_layers"],
                use_topk_query=self.model_cfg["use_topk_query"],
                use_msf=self.model_cfg["use_msf"],
                dec_layers=self.model_cfg["dec_layers"],

                # 一些可选的config，即如果文件中没有配置，那么就默认
                **filter_optional_cfg(self.model_cfg, {
                    "dropout", "pool_type",
                    "lstm_layers", "lstm_drop",
                    "char_lstm_layers", "char_lstm_drop",
                    "fpn_drop",
                    "dec_intermediate_size", "dec_num_attention_heads"
                })
            )
        else:
            # create model
            self.config = plm_cfg_cls.from_pretrained(self.model_cfg["plm_path"])
            self.model = self.MODEL_CLASS(
                config=self.config,
                n_classes=task.num_entity_types,
                n_queries=self.model_cfg["num_entity_queries"],
                use_lstm=self.model_cfg["use_lstm"],
                token_ebd=token_ebd,
                pos_ebd=pos_ebd,
                char_ebd=char_ebd,

                fpn_type=self.model_cfg["fpn_type"],
                fpn_layers=self.model_cfg["fpn_layers"],
                use_topk_query=self.model_cfg["use_topk_query"],
                use_msf=self.model_cfg["use_msf"],
                dec_layers=self.model_cfg["dec_layers"],

                # 一些可选的config，即如果文件中没有配置，那么就默认
                **filter_optional_cfg(self.model_cfg, {
                    "dropout", "pool_type",
                    "lstm_layers", "lstm_drop",
                    "char_lstm_layers", "char_lstm_drop",
                    "fpn_drop",
                    "dec_intermediate_size", "dec_num_attention_heads"
                })
            )
            self.model.init()

            self.module.fpn.encoder.with_pretrained(path=self.model_cfg["plm_path"])

        # 这个只有bp的时候用到，不算模型参数，记得to device
        self.left_boundary, self.right_boundary = generate_boundary(
            self.config.max_position_embeddings, self.model_cfg["fpn_layers"] + 1
        )

        if self.criterion_cfg is not None:
            # inference的时候可能没有这些东西
            stage2_criterion_cfg = self.criterion_cfg["stage2"]
            self.soft_boundary = None
            if stage2_criterion_cfg.get("use_soft_proposal_boundary_matching", False):
                sigma = stage2_criterion_cfg["soft_boundary_sigma"]
                token_indexes = torch.arange(0, self.max_positions(), 1)
                self.soft_boundary = torch.stack([
                    torch.distributions.normal.Normal(
                        torch.as_tensor([i]), torch.as_tensor([sigma])
                    ).log_prob(token_indexes).exp()
                    for i in range(self.max_positions())
                ])
            self.stage2_criterion = NerSetCriterion(
                match_weight=stage2_criterion_cfg["match_weight"],
                **filter_optional_cfg(
                    stage2_criterion_cfg,
                    {"nil_weight", "match_solver", "boundary_loss_type", "cls_loss_type"}
                )
            )

    def max_positions(self):
        """Maximum length supported by the model."""
        return self.config.max_position_embeddings

    def optim_params(self, **kwargs):
        return prepare_trf_based_model_params(
            self.model,
            self.module.fpn.encoder.trf,
            **filter_optional_cfg(kwargs, {"weight_decay", "trf_lr"}),
        )

    def set_requires_grad(self, requires_grad: bool, mode: str, **kwargs):
        if mode == "fpn.encoder.trf":
            for _, param in self.module.fpn.encoder.trf.named_parameters():
                param.requires_grad = requires_grad
        elif mode == "fpn":
            for _, param in self.module.fpn.named_parameters():
                param.requires_grad = requires_grad
        elif mode == "fpn+stage1head":
            for module in [
                self.module.fpn,
                self.module.padding_span_feature, self.module.feat_trans, self.module.pos_feat_norm, self.module.cls_feat_norm,
                self.module.stage1_classifier
            ]:
                for _, param in module.named_parameters():
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
        htokens, stage1_output, stage2_output = self.model.forward(
            encodings=batch["encoding"],
            encoding_masks=batch["encoding_masks"],
            token2encoding_masks=batch["token2encoding_masks"],
            token2start=batch["token2start"],
            token_masks=batch['token_masks'],
            pos_encoding=batch['pos_encoding'],
            w2v_encoding=batch["w2v_encoding"],
            char_encoding=batch['char_encoding'],
            token_masks_char=batch['token_masks_char'],
            char_count=batch['char_count'],
        )
        bsz, len_seq, _ = htokens.size()

        outputs = {
            "stage1": stage1_output,
            "stage2": stage2_output,
        }

        if needs_loss:
            # stage1 boundary，inference是不需要的，只有在计算loss的时候需要
            if stage1_output is not None:
                flat_padding_masks = stage1_output["flat_padding_masks"]
                _, candidates = flat_padding_masks.size()

                def make_boundary(boundary_matrix: Tensor):
                    return torch.cat(
                        [boundary_matrix[i, :sz[1]] for i, sz in enumerate(stage1_output["layer_size"])], dim=-1
                    ).expand(bsz, -1)

                left_boundary = make_boundary(self.left_boundary).to(flat_padding_masks.device)
                right_boundary = make_boundary(self.right_boundary).to(flat_padding_masks.device)

                left_boundary = torch.masked_fill(left_boundary, flat_padding_masks, value=len_seq - 1)
                right_boundary = torch.masked_fill(right_boundary, flat_padding_masks, value=0)

                # NOTE: 为了和predictor的输出一致，这里将左右边界设为1，其余位置设为-1e25
                proposal_left = torch.full(
                    (bsz, candidates, len_seq), fill_value=-1e25,
                    dtype=torch.float, device=flat_padding_masks.device
                )
                proposal_left = torch.scatter(proposal_left, dim=-1, index=left_boundary.unsqueeze(-1), value=1.)
                proposal_right = torch.full(
                    (bsz, candidates, len_seq), fill_value=-1e25,
                    dtype=torch.float, device=flat_padding_masks.device
                )
                proposal_right = torch.scatter(proposal_right, dim=-1, index=right_boundary.unsqueeze(-1), value=1.)
                stage1_output["proposal_left"] = proposal_left
                stage1_output["proposal_right"] = proposal_right

            # print(batch["raw_sample"])
            stage1_criterion_cfg = self.criterion_cfg["stage1"]
            stage2_criterion_cfg = self.criterion_cfg["stage2"]
            sub_losses = []
            sub_loss_weights = []

            if stage1_output is not None:
                #第一阶段的loss
                sub_losses.append(self._stage1_loss(batch, stage1_output))
                sub_loss_weights.append(stage1_criterion_cfg["loss_weight"])

            # 第二阶段的loss
            set_loss_weight = stage2_criterion_cfg["loss_weight"]
            if stage2_criterion_cfg["deeply_weight"] == "same":
                deeply_weight = [1] * len(stage2_output)
            elif stage2_criterion_cfg["deeply_weight"] == "linear":
                deeply_weight = list(range(1, len(stage2_output) + 1))
            elif stage2_criterion_cfg["deeply_weight"] == "last":
                deeply_weight = [0] * len(stage2_output)
                deeply_weight[-1] = 1
            stage2_loss = self._stage2_loss(batch, stage1_output, stage2_output)

            for lay_i, (w, l) in enumerate(zip(deeply_weight, stage2_loss)):
                for k, v in l.items():
                    sub_losses.append(v)
                    sub_loss_weights.append(w * set_loss_weight[k])

            outputs["loss"] = self.backward.backward(sub_losses, sub_loss_weights, requires_grad)

        return outputs

    def _stage1_loss(self, batch: Dict[str, Any], stage1_output: Dict[str, Any]) -> Tensor:
        stage1_criterion_cfg = self.criterion_cfg["stage1"]
        nil_weight = stage1_criterion_cfg["nil_weight"]
        neg_ratio = stage1_criterion_cfg.get("neg_ratio", -1)

        stage1_flat_logits = stage1_output["flat_logits"]
        flat_padding_masks = stage1_output["flat_padding_masks"]     # bsz, n_pred
        layer_size = stage1_output["layer_size"]
        flat_masks = ~flat_padding_masks
        n_fpn_layers = len(layer_size)
        gt_types = batch["gt_types"]
        gt_spans = batch["gt_spans"]
        positive_masks = gt_types != 0
        num_pos_spans = torch.sum(positive_masks)

        gts = [torch.zeros(sz[:-1], dtype=gt_types.dtype, device=gt_types.device) for sz in layer_size]
        # NOTE: 注意这里的gt中是可能包含负样本的，因此需要构造gt_masks而不是直接用positive_masks
        gt_masks = [torch.zeros(sz[:-1], dtype=torch.bool, device=gt_types.device) for sz in layer_size]

        for sample_i, (sample_gt_spans, sample_gt_types, sample_mask) in enumerate(
            zip(gt_spans, gt_types, batch["gt_masks"])
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

        flat_pred_logits = torch.masked_select(
            stage1_flat_logits, mask=flat_masks.unsqueeze(-1)
        ).view(-1, stage1_flat_logits.size()[-1])
        flat_gts = torch.masked_select(gts, mask=flat_masks)

        weight = torch.ones(
            stage1_flat_logits.size()[-1],
            dtype=stage1_flat_logits.dtype,
            device=stage1_flat_logits.device
        )
        if nil_weight < 0:
            if flat_gts.size(0) > num_pos_spans:
                # NOTE: 这里做了一个修正，本来cond只有self.nil_weight < 0，
                # 但是在n_gts很大的时候，甚至大于等于n_candidates的时候，原来的公式会计算出inf或者负的weight，这是不合理的
                # 考虑到这个weight是在nil太多的时候（n_candidates << n_gts)平衡nil的权重，而nil比较小的时候，权重和其他（1.0）一样就好了
                weight[0] = num_pos_spans / (flat_gts.size(0) - num_pos_spans)
        else:
            weight[0] = nil_weight
        stage1_loss = F.cross_entropy(flat_pred_logits, flat_gts, weight=weight)

        return stage1_loss

    def _stage2_loss(
        self,
        batch: Dict[str, Any],
        stage1_output: Dict[str, Any],
        stage2_output: Dict[str, Any],
    ) -> List[Dict[str, Tensor]]:
        layer_losses = []

        gt_spans = batch["gt_spans"]            # bsz, n_gt, 2
        gt_types = batch["gt_types"]            # bsz, n_gt
        gt_masks = batch["gt_masks"] & (batch["gt_types"] != 0)     # NOTE: 注意二部图匹配不需要负样本，会自动将所有未匹配的logits对应的gt设为0

        soft_left_boundary = None
        soft_right_boundary = None
        if self.soft_boundary is not None:
            topk_indexes = stage1_output["topk_index"]
            proposal_left = torch.softmax(stage1_output["proposal_left"], dim=-1)
            proposal_right = torch.softmax(stage1_output["proposal_right"], dim=-1)
            proposal_left = torch.gather(
                proposal_left, 1, topk_indexes.unsqueeze(-1).expand(-1, -1, proposal_left.size(-1))
            )
            proposal_right = torch.gather(
                proposal_right, 1, topk_indexes.unsqueeze(-1).expand(-1, -1, proposal_right.size(-1))
            )           # bsz, k, sent_len

            soft_boundary = self.soft_boundary[:proposal_right.size(-1), :proposal_right.size(-1)].to(gt_spans.device)
            soft_left_boundary = torch.einsum("bks,ss->bks", proposal_left, soft_boundary)
            soft_right_boundary = torch.einsum("bks,ss->bks", proposal_right, soft_boundary)

        for raw_layer_outputs in stage2_output:
            outputs = {
                "pred_logits": raw_layer_outputs["entity_logits"],
                "pred_left": raw_layer_outputs["p_left"],
                "pred_right": raw_layer_outputs["p_right"],
                "token_mask": batch["token_masks"],
            }
            if soft_left_boundary is not None and soft_right_boundary is not None:
                outputs["match_left"] = soft_left_boundary
                outputs["match_right"] = soft_right_boundary
            loss_dict, _ = self.stage2_criterion.forward(
                outputs=outputs,
                targets={"gt_types": gt_types, "gt_spans": gt_spans, "gt_masks": gt_masks}
            )

            layer_losses.append(loss_dict)

        return layer_losses
