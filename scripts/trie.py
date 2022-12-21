from typing import List, Optional



class TrieChar:
    """
    中文用的字典树，树中的每一个节点是一个汉字
    """

    class TrieNode:
        """在中文语料下使用，一个节点是一个字符

        """

        def __init__(self, ch: Optional[str], index: Optional[int]):
            self.ch = ch
            self.index = index
            self.children = {}

        def __contains__(self, item):
            return item in self.children

        def __getitem__(self, item):
            return self.children[item]

        def __setitem__(self, key, value):
            self.children[key] = value

        def get_or_create(self, ch: str, case_sensitive: bool = True):
            try_ch = self.try_get(ch, case_sensitive)
            if try_ch is not None:
                return try_ch
            else:
                ret = TrieChar.TrieNode(ch, None)
                self.children[ch] = ret
                return ret

        def try_get(self, key: str, case_sensitive: bool = True):
            for k, v in self.children.items():
                v: TrieChar.TrieNode = v
                if case_sensitive and k == key:
                    return v
                elif not case_sensitive and k.lower() == key.lower():
                    return v
            return None

    def __init__(self):
        self.tree = TrieChar.TrieNode(None, None)
        self.vocab = []

    def __len__(self):
        return len(self.vocab)

    def add_word(self, word: str, case_sensitive: bool = True) -> int:
        chars = list(word)
        tree_node = self.tree
        for ch in chars:
            tree_node = tree_node.get_or_create(ch, case_sensitive)
        if tree_node.index is None:
            # 不存在
            tree_node.index = len(self.vocab)
            self.vocab.append(word)

        return tree_node.index

    def get_word_by_idx(self, idx: int) -> str:
        return self.vocab[idx]

    def startswith(self, s: str, index: int = 0, case_senitive: bool = True) -> List[int]:
        """_summary_

        Args:
            s (str): 
            index (int, optional): 从tokens的第几个开始匹配. Defaults to 0.
            case_senitive (bool, optional): _description_. Defaults to True.

        Returns:
            List[int]: 例如[7, 9]表示tokens[index: 7], tokens[index: 9]都是trie中的词，list是递增的
        """
        ret = []
        tree_node = self.tree
        for i, ch in enumerate(s[index:]):
            has_match = True
            try_sub_tree = tree_node.try_get(ch, case_senitive)
            if try_sub_tree is not None:
                tree_node = try_sub_tree
            else:
                has_match = False
                break
            if has_match:
                if tree_node.index is not None:
                    # 存在一个词
                    ret.append(index + i + 1)  # return tail
            else:
                break
        return ret

    def contains(self, s: str, case_sensitive: bool = True) -> bool:
        chars = list(s)
        tree_node = self.tree
        for ch in chars:
            tree_node = tree_node.try_get(ch, case_sensitive)
            if tree_node is None:
                return False
        return tree_node.index is not None


class Trie:

    class TrieNode:

        def __init__(self, token: Optional[str], index: Optional[int]):
            self.token = token
            self.index = index
            self.children = {}

        def __contains__(self, item):
            return item in self.children

        def __getitem__(self, item):
            return self.children[item]

        def __setitem__(self, key, value):
            self.children[key] = value

        def get_or_create(self, token: str, case_sensitive: bool = True):
            try_token = self.try_get(token, case_sensitive)
            if try_token is not None:
                return try_token
            else:
                ret = Trie.TrieNode(token, None)
                self.children[token] = ret
                return ret

        def try_get(self, key: str, case_sensitive: bool = True):
            for k, v in self.children.items():
                v: Trie.TrieNode = v
                if case_sensitive and k == key:
                    return v
                elif not case_sensitive and k.lower() == key.lower():
                    return v
            return None

    def __init__(self):
        self.tree = Trie.TrieNode(None, None)
        self.vocab = []

    def __len__(self):
        return len(self.vocab)

    def add_word(self, word: List[str], case_sensitive: bool = True) -> int:
        tokens = list(word)
        tree_node = self.tree
        for tok in tokens:
            tree_node = tree_node.get_or_create(tok, case_sensitive)
        if tree_node.index is None:
            # 不存在
            tree_node.index = len(self.vocab)
            self.vocab.append(word)

        return tree_node.index

    def get_word_by_idx(self, idx: int) -> str:
        return self.vocab[idx]

    def startswith(self, tokens: List[str], case_senitive: bool = True) -> List[int]:
        """_summary_

        Args:
            tokens (List[str]): list of words.
            case_senitive (bool, optional): _description_. Defaults to True.

        Returns:
            List[int]: 例如[7, 9]表示tokens[index: 7], tokens[index: 9]都是trie中的词，list是递增的
        """
        ret = []
        tree_node = self.tree
        for i, tok in enumerate(tokens):
            try_sub_tree = tree_node.try_get(tok, case_senitive)
            if try_sub_tree is not None:
                tree_node = try_sub_tree
                if tree_node.index is not None:
                    # 存在一个词
                    ret.append(i + 1)  # return tail
            else:
                break
        return ret

    def contains(self, s: List[str], case_sensitive: bool = True) -> bool:
        tree_node = self.tree
        for tok in s:
            tree_node = tree_node.try_get(tok, case_sensitive)
            if tree_node is None:
                return False
        return tree_node.index is not None
