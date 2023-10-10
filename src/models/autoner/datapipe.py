
from typing import Any, Dict, Iterable, Iterator
from alchemy.pipeline import DataPipeline, ItrDataPipeline
from alchemy import sym_tbl
import numpy as np
from ...task.ner import NerTask
from ...task.ner.entities import Sample, Mention, Token, EntityType


@DataPipeline.register()
class SampleWithAutoNerLabel(ItrDataPipeline):
    def __init__(self, datapipe: ItrDataPipeline, **kwargs):
        super().__init__()
        self.datapipe = datapipe

    def __iter__(self) -> Iterator:
        for d in self.datapipe:
            yield self.tagging(d)

    @staticmethod
    def tagging(sample: Dict[str, Any]):
        raw_sample: Sample = sample["raw_sample"]
        entity_types: Dict[str, EntityType] = sym_tbl().task.entity_types

        # 0 for break，1 for tie
        chunk_label = np.ones(len(raw_sample.tokens), dtype=int)
        type_label = np.zeros(len(raw_sample.tokens), dtype=int)
        type_mask = np.zeros(len(raw_sample.tokens), dtype=bool)
        chunk_label[0] = 0
        type_mask[0] = True
        for m in raw_sample.mentions:
            e_start, e_end, ty = m.as_tuple_token()
            e_end = e_end + 1       # end is inclusive
            chunk_label[e_start] = 0        # 0 for break
            type_label[e_start] = ty.index
            type_mask[e_start] = True
            if e_end < len(chunk_label):
                chunk_label[e_end] = 0
                type_mask[e_end] = True

        sample["chunk_label"] = chunk_label
        sample["type_label"] = type_label
        sample["type_mask"] = type_mask

        return sample