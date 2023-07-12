import collections


def convert_to_unicode(text):
    """
    Convert `text` to Unicode (if it's not already), assuming utf-8 input.
    """
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


class Vocabulary(object):
    """
    Vocabulary class.
    """

    def __init__(self, vocab_file):
        self.vocab = self.load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.num_relations = len(self.vocab) - 2
        # 2 for special tokens of [PAD] and [MASK]

    def load_vocab(self, vocab_file):
        """
        Load a vocabulary file into a dictionary.
        """
        vocab = collections.OrderedDict()
        fin = open(vocab_file)
        for num, line in enumerate(fin):
            items = convert_to_unicode(line.strip()).split("\t")
            if len(items) > 2:
                raise
            token = items[0]
            index = num * 2
            token = token.strip()           
            vocab[token] = int(index)

            #id of inverse relations are odds
            token_inv = 'inv_' + token
            index_inv = num * 2 + 1
            token_inv = token_inv.strip()
            vocab[token_inv] = int(index_inv)

        return vocab

    def convert_by_vocab(self, vocab, items):
        """
        Convert a sequence of [tokens|ids] using the vocab.
        """
        output = []
        for item in items:
            output.append(vocab[item])
        return output

    def convert_tokens_to_ids(self, tokens):
        return self.convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return self.convert_by_vocab(self.inv_vocab, ids)