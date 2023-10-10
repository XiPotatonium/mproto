"""
TODO: 这里需要重构一下，span和range需要区分开，否则容易产生混淆，顺便全部改成dataclass好了
Token其实应该是Word，存在概念的混淆
get_span_tokens的实现很奇怪，是不是不需要这个东西？
Word - SubWord - Char这三层结构，不要用token这个词，因为token只有在输入模型的时候有意义，在数据中无意义，反而会引入歧义
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class EntityType:
    identifier: str
    index: int
    short_name: str
    verbose_name: str

    def __eq__(self, other):
        if isinstance(other, EntityType):
            return self.identifier == other.identifier
        return False

    def __hash__(self):
        return hash(self.identifier)


# Word not token
class Token:
    # POS_MAP = [
    #   "ADJ", "ADP", "ADV", "AUX", "CONJ", "CCONJ", "DET", "INTJ",
    #   "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE"
    # ]
    def __init__(
            self,
            index: int,
            span_start: int,
            span_end: int,
            phrase: str,
            pos: int,
            vocab_id: int,
            char_start: int,
            char_end: int
    ):
        self.index = index  # original token index in document, i-th token in the sentence

        self.span_start = span_start  # start of token span (BPE) in document (inclusive)
        # TODO: 这里有歧义，exclusive应该重命名为range才对
        self.span_end = span_end  # end of token span (BPE) in document (exclusive)
        self.char_start = char_start
        self.char_end = char_end
        self.phrase = phrase
        self.pos = pos
        self.vocab_id = vocab_id

    @property
    def wordinx(self):
        return self.vocab_id

    @property
    def span(self):
        return self.span_start, self.span_end

    @property
    def char_span(self):
        return self.char_start, self.char_end

    @property
    def pos_id(self):
        # return self.POS_MAP.index(self._pos)
        return self.pos

    def __repr__(self) -> str:
        return self.phrase


class TokenSpan:
    def __init__(self, tokens: List[Token]):
        self._tokens = tokens

    @property
    def span_start(self) -> int:
        return self._tokens[0].span_start

    @property
    def span_end(self) -> int:
        return self._tokens[-1].span_end

    @property
    def span(self) -> Tuple[int, int]:
        return self.span_start, self.span_end

    # @property
    # def c(self):
    #     return self._tokens[0].index,self._tokens[-1].index + 1

    def __getitem__(self, s):
        if isinstance(s, slice):
            return TokenSpan(self._tokens[s.start:s.stop:s.step])
        else:
            return self._tokens[s]

    def __iter__(self):
        return iter(self._tokens)

    def __len__(self):
        return len(self._tokens)


class Mention:
    def __init__(self, entity_type: EntityType, tokens: List[Token]):
        self.entity_type = entity_type

        self._tokens = tokens

    def as_tuple(self):
        return self.span_start, self.span_end, self.entity_type

    def as_tuple_token(self):
        return self._tokens[0].index, self._tokens[-1].index, self.entity_type

    def as_token_span(self):
        return TokenSpan(self._tokens)

    @property
    def tokens(self):
        return TokenSpan(self._tokens)

    @property
    def span_start(self):
        return self._tokens[0].span_start

    @property
    def span_end(self):
        return self._tokens[-1].span_end

    @property
    def span(self):
        return self.span_start, self.span_end

    @property
    def span_token(self):
        """
        注意两边都是inclusive的

        Returns:

        """
        return self._tokens[0].index, self._tokens[-1].index


@dataclass
class Sample:
    id: str
    tokens: List[Token]
    mentions: List[Mention]
    encoding: List[int]
    char_encodings: List[List[int]]
    seg_encoding: List[int]


def get_span_tokens(tokens, span):
    inside = False
    span_tokens = []

    for t in tokens:
        # print(t.index)
        if t.index == span[0]:
            inside = True

        if inside:
            span_tokens.append(t)

        if inside and t.index == span[1]:
            return TokenSpan(span_tokens)

    if span[0] < 0 or span[0] >= len(tokens) or span[1] < 0 or span[1] >= len(tokens):
        raise ValueError("span {} not within tokens of length {}".format(span, len(tokens)))
    return None # 有可能left > right
