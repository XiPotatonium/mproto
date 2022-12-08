from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List, Tuple

import numpy as np

from .entities import Mention

class TaggingScheme(ABC):
    def __init__(self, entity_types: OrderedDict, idx_padding: int = -100):
        if not isinstance(entity_types, OrderedDict):
            raise ValueError()
        self.entity_types = entity_types
        self.idx_padding = idx_padding

    @abstractmethod
    def decode_tags(self, tags: np.ndarray) -> List[Tuple[int, int, int]]:
        """注意这里得到的是range，也就是说右边界是exclusive的

        Args:
            tags (np.ndarray): _description_

        Returns:
            List[Tuple[int, int, int]]: _description_
        """
        pass

    @abstractmethod
    def encode_tags(self, entities: List[Mention], sent_lens: int, dtype=int) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def num_tags(self) -> int:
        pass

class BioTaggingScheme(TaggingScheme):
    def __init__(self, entity_types: OrderedDict, idx_padding: int = -100):
        super().__init__(entity_types, idx_padding)
        # O的index = 0
        # B-XX的index = entity_type.index * 2 - 1
        # I-XX的index = entity_type.index * 2
        # 注意看BaseNerTask的构造函数，这里的None的index是0
        self.idx_O = 0
        if 0 <= idx_padding <= self.num_tags:
            raise ValueError("Invalid {}".format(idx_padding))

    def decode_tags(self, tags: np.ndarray) -> List[Tuple[int, int, int]]:
        """注意这里得到的是range，也就是说右边界是exclusive的

        Args:
            tags (np.ndarray): _description_

        Returns:
            List[Tuple[int, int, int]]: _description_
        """
        ret = []
        last_b_idx = -1
        for i, tag in enumerate(tags):
            if tag == self.idx_O:
                if last_b_idx >= 0:
                    # 一个span结尾
                    ret.append((last_b_idx, i, (tags[last_b_idx] + 1) // 2))
                    last_b_idx = -1
            elif tag % 2 == 1:
                # B tag
                if last_b_idx >= 0:  # 一个span结尾
                    ret.append((last_b_idx, i, (tags[last_b_idx] + 1) // 2))
                    last_b_idx = -1
                last_b_idx = i
            else:
                # I tag
                if last_b_idx >= 0:
                    if tag - 1 == tags[last_b_idx]:
                        # consistent
                        pass
                    else:
                        # inconsistent
                        ret.append((last_b_idx, i, (tags[last_b_idx] + 1) // 2))
                        last_b_idx = -1
        if last_b_idx >= 0:
            ret.append((last_b_idx, len(tags), (tags[last_b_idx] + 1) // 2))

        return ret

    def encode_tags(self, entities: List[Mention], sent_lens: int, dtype=int) -> np.ndarray:
        tags = np.ones(sent_lens, dtype=dtype) * self.idx_O
        for e in entities:
            e_start, e_end = e.span_token       # 注意e_end是inclusive
            if e.entity_type.index == 0:
                continue            # 不要标记O type，因为O没有B和I
            tags[e_start] = e.entity_type.index * 2 - 1
            if e_end > e_start:
                tags[e_start + 1:e_end + 1] = e.entity_type.index * 2
        return tags

    @property
    def num_tags(self):
        """除了None只有O之外，其他都是B-XX和I-XX

        Returns:
            _type_: _description_
        """
        return len(self.entity_types) * 2 - 1


class IOTaggingScheme(TaggingScheme):
    def __init__(self, entity_types: OrderedDict, idx_padding: int = -100):
        super().__init__(entity_types, idx_padding)
        # I-XX的index = entity_type.index
        # 注意看BaseNerTask的构造函数，这里的None的index是0
        self.idx_O = 0
        if 0 <= idx_padding <= self.num_tags:
            raise ValueError("Invalid {}".format(idx_padding))

    def decode_tags(self, tags: np.ndarray) -> List[Tuple[int, int, int]]:
        """注意这里得到的是range，也就是说右边界是exclusive的

        Args:
            tags (np.ndarray): _description_

        Returns:
            List[Tuple[int, int, int]]: _description_
        """
        ret = []
        last_b_idx = -1
        for i, tag in enumerate(tags):
            if tag == self.idx_O:
                if last_b_idx >= 0:
                    # 一个span结尾
                    ret.append((last_b_idx, i, tags[last_b_idx]))
                    last_b_idx = -1
            else:
                # I tag
                if last_b_idx >= 0:
                    if tag == tags[last_b_idx]:
                        # consistent
                        pass
                    else:
                        # inconsistent
                        ret.append((last_b_idx, i, tags[last_b_idx]))
                        last_b_idx = i
                else:
                    last_b_idx = i
        if last_b_idx >= 0:
            ret.append((last_b_idx, len(tags), tags[last_b_idx]))

        return ret

    def encode_tags(self, entities: List[Mention], sent_lens: int, dtype=int) -> np.ndarray:
        tags = np.ones(sent_lens, dtype=dtype) * self.idx_O
        for e in entities:
            e_start, e_end = e.span_token       # 注意e_end是inclusive
            tags[e_start:e_end + 1] = e.entity_type.index
        return tags

    @property
    def num_tags(self):
        """num_entity_types个I，一个O

        Returns:
            _type_: _description_
        """
        return len(self.entity_types)
