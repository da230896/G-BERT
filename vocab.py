"""
    Basic vocab class that will be used for storing medication and diseases codes:
"""

class Vocab:
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}

    def add_sentence(self, sentence: list[str]):
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.idx2word)] = word
                self.word2idx[word] = len(self.word2idx)

    def add_special_tokens(self):
        self.add_sentence(("[PAD]", "[MASK]", "[CLS]", "[UNK]"))
        # Currently I am not using [PAD] tokens since they are masked using seq_mask
        # TODO: Need to add [MASK] in pre-training