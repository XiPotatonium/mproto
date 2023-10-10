import json
from collections import OrderedDict
from pathlib import Path
import shutil
import string
from typing import Any, Dict, List, Optional

import numpy as np

from loguru import logger
from transformers import BertTokenizer, RobertaTokenizer, PreTrainedTokenizer, AutoTokenizer

from alchemy import AlchemyTask, sym_tbl
from alchemy.pipeline import DataPipeline, OutputPipeline, EvalPipeline
from .ner_scorer import NerScorer
from .. import CollateFn, DefaultCollateFn
from ...util.vocab import build_dictionary, build_pos_dictionary, strip_vocab
from .entities import EntityType


@AlchemyTask.register("NerTask")
class NerTask(AlchemyTask):
    def __init__(self):
        super(NerTask, self).__init__()
        record_dir: Optional[Path] = sym_tbl().try_get_global("record_dir")
        meta_path = Path(self.cfg["meta"])
        with meta_path.open('r', encoding="utf8") as f:
            meta_info = json.load(f, object_pairs_hook=OrderedDict)

        self.entity_types = OrderedDict()

        # entities
        # add 'None' entity type
        none_entity_type = EntityType('None', 0, 'None', 'No Entity')
        self.entity_types['None'] = none_entity_type

        # specified entity types
        for i, (key, v) in enumerate(meta_info['entities'].items()):
            if key == "None":
                # 一般而言不会重载None
                logger.warning("Entity set {} overrides default None entity type".format(meta_info['entities']))
                none_entity_type.short_name = v['short']
                none_entity_type.verbose_name = v['verbose']
            else:
                entity_type = EntityType(key, len(self.entity_types), v['short'], v['verbose'])
                self.entity_types[key] = entity_type

        self.idx2entity_type = {t.index: t for t in self.entity_types.values()}

        # dictionaries
        self.pos_vocab= strip_vocab(None, [], special_tokens=["<UNK>"],)
        if sym_tbl().cfg["model"].get("use_pos", False):
            self.pos_vocab, pos_vocab_cache_path = build_pos_dictionary(
                corpus_paths=self.cfg.get("corpus"),
                cache_dir=self.cfg.get("vocab_cache_dir", str(meta_path.parent)),
                cache_tag=self.cfg.get("vocab_cache_tag"),
                special_tokens=["<UNK>"],
            )
            logger.info(f"Use pos vocab with size = {len(self.pos_vocab)}")
            if record_dir is not None:
                # 备份vocab，保证可复现，vocab不会很大
                shutil.copy(str(pos_vocab_cache_path), str(record_dir))

        self.token_vocab= strip_vocab(None, [], special_tokens=["<UNK>"],)
        self.token_ebd: Optional[np.ndarray] = None
        if sym_tbl().cfg["model"].get("use_w2v", False):
            if "w2v_path" not in sym_tbl().cfg["model"]:
                raise Exception()
            self.token_vocab, token_vocab_cache_path, self.token_ebd, _ = build_dictionary(
                filename=sym_tbl().cfg["model"]["w2v_path"],
                corpus_paths=self.cfg.get("corpus"),
                cache_dir=self.cfg.get("vocab_cache_dir", str(meta_path.parent)),
                cache_tag=self.cfg.get("vocab_cache_tag"),
                special_tokens=["<UNK>"],
                ignore_missing_ebd_cache=self.cfg.get("ignore_missing_ebd_cache", False),
            )
            logger.info(f"Use token vocab with size = {len(self.token_vocab)}")
            if record_dir is not None:
                # 只备份vocab，因为ebd包含在模型参数中
                shutil.copy(str(token_vocab_cache_path), str(record_dir))

        # NOTE: 现在只考虑了printable
        self.char_vocab = strip_vocab(None, list(string.printable), special_tokens=['<PAD>', '<EOT>', '<UNK>'])

        self.tokenizer = self.build_tokenizer()

        self.scorer = NerScorer(self.entity_types)

        for p_cfg in self.cfg.get("outputpipes", []):
            self.outputpipes.append(
                # 注意这个时候ctx.task还是None，手动传一下task吧
                OutputPipeline.from_registry(p_cfg["type"], **p_cfg)
            )

        for p_cfg in self.cfg.get("evalpipes", []):
            self.evalpipes.append(
                # 注意这个时候ctx.task还是None，手动传一下task吧
                EvalPipeline.from_registry(p_cfg["type"], **p_cfg)
            )

    @property
    def num_entity_types(self):
        return len(self.entity_types)

    def max_positions(self):
        return super().max_positions()

    def build_tokenizer(self) -> PreTrainedTokenizer:
        return AutoTokenizer.from_pretrained(
            sym_tbl().cfg["model"]["tokenizer_path"],
            local_files_only=True,
            do_lower_case=sym_tbl().cfg["model"]["lowercase"],
        )

    def load_dataset(self, split: str, **kwargs):
        logger.info(f"Load dataset {split}")

        if "batch_size" not in kwargs:
            kwargs["batch_size"] = None
        pipes: List[Dict[str, Any]] = kwargs.pop("pipes")
        if "collate_fn" in kwargs:
            collate_fn_cfg = kwargs.pop("collate_fn")
            collate_fn = CollateFn.from_registry(collate_fn_cfg["type"], **collate_fn_cfg)
        else:
            collate_fn = DefaultCollateFn()

        for i, p_cfg in enumerate(pipes):
            if i == 0:
                datapipe = DataPipeline.from_registry(p_cfg["type"], **p_cfg)
            else:
                datapipe = DataPipeline.from_registry(p_cfg["type"], datapipe, **p_cfg)

        self._datasets[split] = (datapipe, {"collate_fn": collate_fn, **kwargs})
