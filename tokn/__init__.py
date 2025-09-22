from typing import Any, Dict, Generator, Iterable, List, Tuple, Union


def ngrams_by_freq(
    samples: Iterable[List[Union[str, int]]], n: int = 1
) -> List[Tuple[Union[str, int], ...]]:
    """Extracts ngrams from the samples in the provided iterable and returns them sorted
     by their frequency of occurrence from more to less frequent

    :param samples: An iterable of sequences, each being a list of strings or integers
     from which to extract n-grams
    :param n: The size of the n-grams to extract. Defaults to 1 (unigrams)

    :return: A list of n-gram tuples sorted by frequency in descending order
     (most frequent first)
    """
    ngram_counts: Dict[Tuple[Union[str, int], ...], int] = {}
    for sample in samples:
        for i in range(0, len(sample) - n + 1):
            ngram = tuple(sample[i : i + n])
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

    return list(
        map(
            lambda x: x[0],
            sorted(ngram_counts.items(), key=lambda x: x[1], reverse=True),
        )
    )


def _bigrams(sample: List[int]) -> Generator[tuple[int, ...], Any, None]:
    """Extract bigrams from a list of integers

    :param sample: sequence of integers
    :return: generator of tuples of integers representing the extracted bigrams
    """
    for i in range(0, len(sample), 2):
        yield tuple(sample[i : i + 2])


class TokenizerTreeNode:
    """Node in the tree of the tokenizer. It has a set of children which are also nodes
    in the tree. V holds the id (int) of the corresponding token string of the path
    followed from the root of the tree down to the current node.
    """

    def __init__(self):
        self.children: Dict[str, TokenizerTreeNode] = {}
        self.v = None


class TokenizerTree:
    """Tokenizer tree to perform a more optimal and efficient tokenization"""

    def __init__(self):
        """Initializes an empty tokenizer tree"""
        self.root: TokenizerTreeNode = TokenizerTreeNode()

    def add(self, k: str, v: int):
        """Adds a new token to the tree

        :param k: string representing the new token
        :param v: id of the token
        :return:
        """
        node = self.root
        # Navigate the tree or create the path to add the token based on the
        # string representation
        for c in list(k):
            if c in node.children:
                node = node.children[c]
            else:
                node.children[c] = TokenizerTreeNode()
                node = node.children[c]
        node.v = v

    def tokenize(self, k: str) -> List[int]:
        """Tokenizes a string using the learned tokenization tree

        :param k: string to be tokenized
        :return: list of token ids
        """
        node = self.root
        tokens = []
        since_last_token = []
        EOS = None
        # Create a sequence of characters followed by the End Of Sequence object.
        # Then reverse it so we can pop from it.
        chars = list(reversed(list(k) + [EOS]))
        # Loop until we go through the whole sequence
        while len(chars) > 0:
            c = chars.pop()
            # If we reach the end of the string and nothing is left
            # to be tokenized, return
            if c == EOS and len(since_last_token) == 0:
                break
            # If the character is a child of the current node, navigate to it,
            # save the char and continue
            if c in node.children:
                node = node.children[c]
                since_last_token.append((c, node))
            else:
                # If the token is not a child of the current node, add it back to the
                # chars sequence and get the token id of the stored sequence up until
                # now if it exists.
                chars.append(c)
                if node.v is not None:
                    tokens.append(node.v)
                    node = self.root
                    since_last_token = []
                else:
                    # If no token matches the sequence stored until now, then start
                    # adding characters back to the main sequence until we find a token
                    # for the remaining stored char list.
                    for ci, ni in reversed(since_last_token):
                        if ni.v is None:
                            chars.append(ci)
                        else:
                            tokens.append(ni.v)
                            node = self.root
                            break
                    since_last_token = []
        return tokens


class Tokn:
    """Tokn BPE tokenizer. This class provides the methods to train and use a BPE
    tokenizer up to the desired vocabulary size.

    """

    def __init__(self, max_vocab: int = 100):
        """Initializes a new BPE tokenizer with a vocabulary up to max_vocab tokens.

        :param max_vocab: maximum number of tokens in the vocabulary
        """
        self.base_map: Dict[str, int] = {}
        self.encode_map: Dict[Tuple[int, int], int] = {}
        self.decode_map: Dict[int, str] = {}
        self.max_vocab = max_vocab

    def _train(self, dataset: Iterable[str]):
        """Trains the BPE tokenizer given a dataset

        :param dataset: sequence of strings used to train the tokenizer
        :return:
        """
        # First we create one token per character in the dataset up to the
        # maximum vocabulary
        count = 0
        for c in ngrams_by_freq((list(sample) for sample in dataset), n=1):
            idx = count
            k = str(c[0])
            self.base_map[k] = idx
            self.decode_map[idx] = k
            count += 1
            if count >= self.max_vocab:
                return

        # Then we build bigrams on the encoded samples to create new tokens.
        # Iterate until the maximum vocabulary size is reached or until no new tokens
        # are created.
        changes = True
        while changes and (count < self.max_vocab):
            changes = False
            for c in ngrams_by_freq(
                (self.simple_encode(sample) for sample in dataset), n=2
            ):
                if c not in self.encode_map:
                    idx = count
                    self.encode_map[c] = idx
                    self.decode_map[idx] = "".join(
                        [self.decode_map[c[0]], self.decode_map[c[1]]]
                    )
                    count += 1
                    changes = True
                    if count >= self.max_vocab:
                        return

    def _build_tree(self):
        """Builds the tree required to use the more efficient tree tokenizer.

        :return:
        """
        self.tree = TokenizerTree()
        # We loop through the items in the learned decode map and add them to the tree.
        for v, k in self.decode_map.items():
            self.tree.add(k, v)

    def train(self, dataset: Iterable[str]):
        """Trains the tokenizer given a sequence of strings

        :param dataset: sequence of strings to train the tokenizer
        :return:
        """
        # First train the tokenizer to learn the encode and decode maps.
        self._train(dataset)
        # Then build the tokenizer tree for a more efficient and optimal encoding.
        self._build_tree()

    def encode(self, s: str) -> List[int]:
        """Encodes a string into a list of token ids (int) using the efficient method of
         traversing a token tree.

        :param s: sequence to be encoded into token ids
        :return: list of token ids (int)
        """
        return self.tree_encode(s)

    def simple_encode(self, s: str) -> List[int]:
        """Encodes a string into a list of tokens using a simple but not efficient
         method. It does not guarantee optimal compression neither efficient encoding.

        :param s: sequence to be encoded into token ids
        :return: list of token ids (int)
        """
        enc = list(
            filter(lambda x: x is not None, (self.base_map.get(c, None) for c in s))
        )
        changes = True
        while changes:
            nenc = []
            for c in _bigrams(enc):
                if c in self.encode_map:
                    nenc.append(self.encode_map.get(c))
                else:
                    nenc.extend(c)
            changes = nenc != enc
            enc = nenc
        return enc

    def tree_encode(self, s: str) -> List[int]:
        """Encodes a string into a list of token ids (int) using the efficient method
         of traversing a token tree.

        :param s: sequence to be encoded into token ids
        :return: list of token ids (int)
        """
        return self.tree.tokenize(s)

    def decode(self, tokens: List[int]) -> str:
        """Decodes a list of token ids (int) and returns the corresponding string.

        :param tokens: list of token ids (int) to be decoded into a string
        :return: the string corresponding to the provided list of token ids
        """
        return "".join([self.decode_map.get(tk, "") for tk in tokens])
