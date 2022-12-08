import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, OrderedDict
from loguru import logger
import numpy as np

from transformers import PreTrainedTokenizer
from alchemy.pipeline import DataPipeline, ItrDataPipeline
from alchemy.util import merge_max_position_constraints
from alchemy import sym_tbl
from .tagging import BioTaggingScheme, TaggingScheme
from ...util import pad_sequence_numpy
from ...util.vocab import Vocab
from . import NerTask
from .entities import Sample, Mention, Token


@DataPipeline.register("JsonLOpener")
class JsonLOpener(ItrDataPipeline):

    def __init__(self, datapipe: Iterable[str], **kwargs):
        super().__init__()
        self.datapipe = datapipe

    def __iter__(self) -> Iterator:
        for file in self.datapipe:
            with Path(file).open('r', encoding="utf8") as f:
                for line in f:
                    yield json.loads(line)


@DataPipeline.register("ParseJsonDoc")
class ParseJsonDoc(ItrDataPipeline):
    def __init__(self, datapipe: ItrDataPipeline, **kwargs):
        super().__init__()
        self.datapipe = datapipe
        task: NerTask = sym_tbl().task
        self.entity_types = task.entity_types
        self.pos_vocab = task.pos_vocab
        self.token_vocab = task.token_vocab
        self.char_vocab = task.char_vocab
        self.tokenizer = task.tokenizer

    def __iter__(self) -> Iterator:
        for d in self.datapipe:
            yield self.read_jsample(
                d,
                self.entity_types,
                self.token_vocab,
                self.pos_vocab,
                self.char_vocab,
                self.tokenizer
            )

    @staticmethod
    def read_jsample(
        jsample: Dict,
        entity_types: OrderedDict,
        token_vocab: Vocab,
        pos_vocab: Vocab,
        char_vocab: Vocab,
        tokenizer: PreTrainedTokenizer,
    ) -> Sample:
        jtokens = jsample['tokens']
        jentities = jsample.get('entities', [])     # inference的时候可能没有entities
        jpos = jsample.get("pos", ["<UNK>"] * len(jtokens))
        ltokens = jsample.get("ltokens", [])
        rtokens = jsample.get("rtokens", [])

        tokens = []

        special_tokens_map = tokenizer.special_tokens_map
        # 开头有一个[CLS]
        encoding = [tokenizer.convert_tokens_to_ids(special_tokens_map['cls_token'])]
        seg_encoding = [0]
        char_encodings = []

        poss = [pos_vocab.word2idx.get(pos, pos_vocab.word2idx["<UNK>"]) for pos in jpos]

        # parse tokens
        for token in ltokens:
            token_encoding = tokenizer.encode(token, add_special_tokens=False)
            encoding += token_encoding
            seg_encoding += [0] * len(token_encoding)

        for i, token in enumerate(jtokens):
            token_encoding = tokenizer.encode(token, add_special_tokens=False)
            token_encoding_char = []
            for c in token:
                if c in char_vocab.word2idx:
                    token_encoding_char.append(char_vocab.word2idx[c])
                else:
                    token_encoding_char.append(char_vocab.word2idx["<UNK>"])
            span_start, span_end = (len(encoding), len(encoding) + len(token_encoding))
            char_start, char_end = (len(char_encodings), len(char_encodings) + len(token_encoding_char))
            # TODO: case sensitive
            if token.lower() in token_vocab.word2idx:
                inx = token_vocab.word2idx[token.lower()]
            else:
                inx = token_vocab.word2idx["<UNK>"]
            tokens.append(Token(i, span_start, span_end, token, poss[i], inx, char_start, char_end))
            encoding += token_encoding
            seg_encoding += [1] * len(token_encoding)
            token_encoding_char += [char_vocab.word2idx['<EOT>']]
            char_encodings.append(token_encoding_char)

        for token in rtokens:
            token_encoding = tokenizer.encode(token, add_special_tokens=False)
            encoding += token_encoding
            seg_encoding += [0] * len(token_encoding)

        encoding += [tokenizer.convert_tokens_to_ids(special_tokens_map['sep_token'])]
        seg_encoding += [0]

        # parse entity mentions
        mentions = []

        for jentity in jentities:
            entity_type = entity_types[jentity['type']]
            start, end = jentity['start'], jentity['end']

            # create entity mention
            mentions.append(Mention(entity_type, tokens[start:end]))

        # create document
        sample = Sample(
            id=jsample.get("id", ""),
            tokens=tokens,
            mentions=mentions,
            encoding=encoding,
            char_encodings=char_encodings,
            seg_encoding=seg_encoding
        )

        return sample


@DataPipeline.register("PruneLongText")
class PruneLongText(ItrDataPipeline):
    def __init__(self, datapipe: ItrDataPipeline, **kwargs):
        super().__init__()
        self.datapipe = datapipe
        self.max_positions = merge_max_position_constraints(
            sym_tbl().task.max_positions(),
            sym_tbl().model.max_positions()
        )

    def __iter__(self) -> Iterator:
        for d in self.datapipe:
            if self.long_text_filter_predicate(d, self.max_positions):
                yield d

    @staticmethod
    def long_text_filter_predicate(sample: Sample, max_positions: Optional[int]) -> bool:
        if max_positions is not None and len(sample.encoding) > max_positions:
            logger.warning("doc_id={} len = {} > {}, doc = {}".format(
                sample.id, len(sample.encoding), max_positions, sample.tokens
            ))
            return False
        return True


@DataPipeline.register("Sample2Encoding")
class Sample2Encoding(ItrDataPipeline):
    def __init__(self, datapipe: ItrDataPipeline, **kwargs):
        super().__init__()
        self.datapipe = datapipe

    def __iter__(self) -> Iterator:
        for d in self.datapipe:
            yield self.encode(d)

    @staticmethod
    def encode(sample: Sample):
        encoding = np.asarray(sample.encoding)
        pos_encoding = np.asarray([t.pos_id for t in sample.tokens])
        w2v_encoding = np.asarray([t.wordinx for t in sample.tokens])

        char_encoding = []
        char_count = []
        for char_encoding_token in sample.char_encodings:
            char_count.append(len(char_encoding_token))
            char_encoding.append(np.asarray(char_encoding_token))
        char_encoding = pad_sequence_numpy(char_encoding)
        token_masks_char: np.ndarray = char_encoding != 0
        char_count = np.asarray(char_count)

        # masking of tokens
        encoding_masks = np.ones(len(encoding), dtype=bool)
        token_masks = np.ones(len(sample.tokens), dtype=bool)
        token2encoding_masks = []
        token2start = np.ones(len(sample.tokens), dtype=int)

        for i, t in enumerate(sample.tokens):
            start, end = t.span
            mask = np.zeros(len(encoding), dtype=bool)
            mask[start:end] = 1
            token2encoding_masks.append(mask)
            token2start[i] = start
        token2encoding_masks = np.stack(token2encoding_masks)

        gt_span_token = []
        gt_entity_types = []
        gt_masks = []

        for mention in sample.mentions:
            gt_span_token.append(mention.span_token)
            gt_entity_types.append(mention.entity_type.index)
            gt_masks.append(True)

        seg_encoding = np.asarray(sample.seg_encoding)

        if len(gt_entity_types) > 0:
            gt_entity_types = np.asarray(gt_entity_types)
            gt_entity_spans_token = np.asarray(gt_span_token)
            gt_masks = np.asarray(gt_masks)
        else:
            gt_entity_types = np.zeros([1], dtype=int)
            gt_entity_spans_token = np.zeros([1, 2], dtype=int)
            gt_masks = np.zeros([1], dtype=bool)

        return {
            "encoding": encoding,
            "pos_encoding": pos_encoding,
            "w2v_encoding": w2v_encoding,

            "char_encoding": char_encoding,
            "char_count": char_count,
            "token_masks_char": token_masks_char,

            "token2encoding_masks": token2encoding_masks,
            "token2start": token2start,
            "encoding_masks": encoding_masks,
            "token_masks": token_masks,

            "seg_encoding": seg_encoding,

            "gt_types": gt_entity_types,
            "gt_spans": gt_entity_spans_token,
            "gt_masks": gt_masks,

            "raw_sample": sample,
        }


@DataPipeline.register("SampleWithTags")
class SampleWithTags(ItrDataPipeline):
    def __init__(self, datapipe: ItrDataPipeline, **kwargs):
        super().__init__()
        self.datapipe = datapipe

    def __iter__(self) -> Iterator:
        scheme: TaggingScheme = sym_tbl().model.tagging_scheme
        for d in self.datapipe:
            yield self.tagging(d, scheme)

    @staticmethod
    def tagging(sample: Dict[str, Any], scheme: TaggingScheme):
        raw_sample: Sample = sample["raw_sample"]
        gt_seq_labels = scheme.encode_tags(raw_sample.mentions, len(raw_sample.tokens), dtype=int)
        sample["gt_seq_labels"] = gt_seq_labels
        return sample
