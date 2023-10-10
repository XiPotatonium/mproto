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
from .model import Tagger



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
