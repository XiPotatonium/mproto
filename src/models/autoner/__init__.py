from pathlib import Path
from typing import Any, Dict, MutableMapping, Optional
from loguru import logger

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss, KLDivLoss, Embedding, BCEWithLogitsLoss
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel
import numpy as np

from alchemy import AlchemyModel, sym_tbl
from alchemy.pipeline import OutputPipeline
from alchemy.util import batch_to_device, filter_optional_cfg
from alchemy.util.optim import prepare_trf_based_model_params
from ...task.ner.tagging import BioTaggingScheme, IOTaggingScheme, TaggingScheme
from ...task.ner import NerTask
from .ner import AutoNerBert
from .object import softCE
from . import utils



@OutputPipeline.register()
class ProcAutoNerOutput(OutputPipeline):
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, outputs: Any, inputs: MutableMapping[str, Any]) -> Any:
        ret = []

        pred_chunk = outputs["pred_chunk"]           # bsz, ntok
        pred_type = outputs["pred_type"]
        if len(pred_type) != 0:
            pred_type = torch.softmax(pred_type, dim=-1)
            pred_score, pred_type = torch.max(pred_type, dim=-1)             # bsz * nchunk
        wcount = torch.sum(inputs['token_masks'].long(), dim=-1).cpu().numpy()

        ind = 0
        for sample_i, (sample_chunks, wcnt) in enumerate(zip(pred_chunk, wcount)):
            sample_outputs = []
            for i, is_chunk_start in enumerate(sample_chunks[:wcnt]):
                if is_chunk_start:
                    chunk_type = pred_type[ind]
                    type_score = pred_score[ind]
                    ind += 1
                    if sample_outputs:
                        sample_outputs[-1]["end"] = i - 1
                    sample_outputs.append(
                        {"start": i, "end": -1, "type": chunk_type.item(), "score": type_score.item()}
                    )
            if len(sample_outputs):
                sample_outputs[-1]["end"] = wcnt - 1
            ret.append(sample_outputs)

        return ret


@AlchemyModel.register("AutoNerTagger")
class AlchemyAutoNerTagger(AlchemyModel):
    MODEL_CLASS = AutoNerBert

    def __init__(self):
        super().__init__()
        task: NerTask = sym_tbl().task
        # scheme和模型绑定而不是和task绑定，所以应该放在模型里
        self.tagging_scheme = IOTaggingScheme(task.entity_types)     # 原型应该用IO tagging比较合适

        if "model_path" in self.model_cfg:
            self.config = AutoConfig.from_pretrained(self.model_cfg["model_path"])
        else:
            # create model
            self.config = AutoConfig.from_pretrained(self.model_cfg["plm_path"])

        if "model_path" in self.model_cfg:
            self.model = self.MODEL_CLASS.from_pretrained(
                self.model_cfg["model_path"],
                config=self.config,

                y_dim=self.model_cfg["label_dim"],
                y_num=len(task.entity_types),
                droprate=self.model_cfg["dropout"],
            )
        else:
            # create model
            self.model = self.MODEL_CLASS(
                config=self.config,

                y_dim=self.model_cfg["label_dim"],
                y_num=len(task.entity_types),
                droprate=self.model_cfg["dropout"],
            )

            logger.info("Use \"{}\" to initialize {} encoder".format(
                self.model_cfg["plm_path"], self.module.encoder.__class__.__name__
            ))
            self.module.encoder = AutoModel.from_pretrained(self.model_cfg["plm_path"])

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

        output = self.model.forward(
            encodings=batch["encoding"],
            encoding_masks=batch["encoding_masks"],
            token2start=batch["token2start"],
        )

        chunk_score = self.model.chunking(output)

        pred_chunk = (chunk_score < self.model_cfg["threshold"]).squeeze(-1)            # 0是break

        # chunk_index = chunk_index.masked_select(pred_chunk).data.cpu()

        outputs = {
            "pred_chunk": pred_chunk.data.cpu(),
            "pred_type": self.model.typing(output, pred_chunk).data.cpu()
        }

        if needs_loss:
            crit_chunk = BCEWithLogitsLoss()
            crit_type = CrossEntropyLoss()

            token_masks = batch['token_masks']
            bp_masks = token_masks.flatten()     # bsz * n_tokens

            chunk_label = batch["chunk_label"]
            type_mask = batch["type_mask"]
            type_label = batch["type_label"]

            flat_chunk_labels = chunk_label.flatten()[bp_masks].float()
            flat_chunk_score = chunk_score.flatten()[bp_masks]
            chunk_loss = crit_chunk(flat_chunk_score, flat_chunk_labels)

            type_score = self.model.typing(output, type_mask)

            flatten_type_label = type_label.masked_select(type_mask)
            type_loss = crit_type(type_score, flatten_type_label)

            losses = [type_loss, chunk_loss]
            loss_weights = [1.0, 1.0]

            outputs["loss"] = self.backward.backward(losses, loss_weights, requires_grad=requires_grad)
        return outputs