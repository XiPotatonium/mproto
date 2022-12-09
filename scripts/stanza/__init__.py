from dataclasses import dataclass
from typing import List

from stanza.protobuf import ParseTree
from stanza.server import CoreNLPClient


@dataclass
class Token:
    """
    替代stanza反序列化的token，因为那个似乎无法在multiprocessing.Queue中传送
    """
    word: str
    pos: str

    @staticmethod
    def from_stanza(tok):
        return Token(word=tok.word, pos=tok.pos)


NOMINAL_TAGS = {
    "NP", "NN", "NNS", "NNP", "NNPS", "NR", "NT"
}

NP_COMPONENTS = {
    "NP", "NN", "NNS", "NNP", "NNPS", "NR", "NT", "DT", "ADJP", "JJ", "JJS"
}


@dataclass
class ChunkResult:
    tokens: List[Token]
    nps: list  # (token begin, token end)

    def get_chunk(self, chunk_idx: int) -> str:
        (chunk_begin, chunk_end) = self.nps[chunk_idx]
        return ''.join([t.word for t in self.tokens[chunk_begin: chunk_end]])

    @property
    def raw_sent(self) -> str:
        return ''.join([t.word for t in self.tokens])


@dataclass
class MyParseTree:
    yieldBeginIndex: int
    yieldEndIndex: int
    value: str
    child: List


class StanzaHelper:
    def __init__(self, client: CoreNLPClient):
        """

        :param client: 仅仅获取读写权限，close和start请在外部调用
        """
        self.client = client

    def chunk(self, text: str) -> List[ChunkResult]:
        ret = []

        doc = self.client.annotate(text)
        for sentence in doc.sentence:
            tokens = sentence.token
            tokens: List[Token] = [Token.from_stanza(t) for t in tokens]

            constituent_tree = sentence.parseTree
            constituent_tree = StanzaHelper._annotate_tree_span(constituent_tree)
            assert constituent_tree.yieldEndIndex == len(tokens)
            nps = []  # (token start, token end)
            StanzaHelper._find_np(constituent_tree, tokens, nps)

            ret.append(ChunkResult(tokens=tokens, nps=nps))
        return ret

    @staticmethod
    def _annotate_tree_span(tree: ParseTree) -> MyParseTree:
        """
        似乎ParseTree并没有记录span的信息，那么这里重新标注span，yieldBeginIndex和yieldEndIndex将作为span
        没有文档，但是我总感觉叶子节点就是一个token

        Args:
            tree (ParseTree): _description_

        Returns:
            MyParseTree: _description_
        """
        tree.yieldBeginIndex = 0
        return StanzaHelper.__annotate_tree_span(tree)

    @staticmethod
    def __annotate_tree_span(tree: ParseTree) -> MyParseTree:
        begin = tree.yieldBeginIndex
        end = begin
        child = []
        if len(tree.child) == 0:
            # 没有孩子，这是一个叶子节点
            # 没有文档，但是我总感觉叶子节点就是一个token
            end += 1
        else:
            for c in tree.child:
                c.yieldBeginIndex = end     # 下一个sibling的begin是这个child的end
                c = StanzaHelper.__annotate_tree_span(c)
                end = c.yieldEndIndex
                child.append(c)
        return MyParseTree(begin, end, tree.value, child)

    @staticmethod
    def _check_np(tree: MyParseTree) -> bool:
        """
        判断一个span在语法上是不是一个np，在CoreNlp的判断的基础上再严格一些，以避免生成过于大的np

        :param tree:
        :return:
        """
        if tree.value not in NOMINAL_TAGS:
            return False

        for child in tree.child:
            if StanzaHelper._is_leaf(child):
                continue
            if child.value not in NP_COMPONENTS:
                return False
        return True

    @staticmethod
    def _is_phrasal(tree: MyParseTree) -> bool:
        """
        Similar to Java CoreNlp Tree.isPhrasal()

        :param tree:
        :return: Return whether this node is a phrasal node or not.
        A phrasal node is defined to be a node which is not a leaf or a preterminal.
        Worded positively, this means that it must have two or more children, or one child that is not a leaf.
        """
        return not (len(tree.child) == 0 or
                    (len(tree.child) == 1 and StanzaHelper._is_leaf(tree.child[0])))

    @staticmethod
    def _is_leaf(tree: MyParseTree) -> bool:
        """
        Similar to Java CoreNlp Tree.isLeaf()

        :param tree:
        :return:
        """
        return len(tree.child) == 0

    @staticmethod
    def _get_phrase(tree: MyParseTree, tokens: List[Token], sep: str = " ") -> str:
        """_summary_

        Args:
            tree (ParseTree): 必须annotate span
            tokens (List[Token]): _description_
            sep (str, optional): _description_. Defaults to " ".

        Returns:
            str: _description_
        """
        return sep.join([t.word for t in tokens[tree.yieldBeginIndex: tree.yieldEndIndex]])

    @staticmethod
    def _find_np(tree: MyParseTree, leaves: list, nps: list):
        begin = tree.yieldBeginIndex
        end = tree.yieldEndIndex

        if StanzaHelper._check_np(tree):
            if begin < end:
                nps.append((begin, end))
                return

        if StanzaHelper._is_phrasal(tree):
            for child in tree.child:
                StanzaHelper._find_np(child, leaves, nps)
