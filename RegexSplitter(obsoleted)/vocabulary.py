from string import printable
from collections import defaultdict


class Vocabulary:
    def __init__(self):
        # assert len(special_tokens) == 4
        # assert len(lcs_tokens) == 7
        # assert len(printable[:-5]) == 95
        # assert len(self.itos) == 106
        special_tokens = ["<unk>", "<pad>", "<sos>", "<eos>"]
        lcs_tokens = ["\x00", "\x01", "\x02", "\x03", "\x04", "\x05", "\x06"]
        self.itos = special_tokens + lcs_tokens + list(printable[:-5])
        self.stoi = defaultdict(lambda: self.stoi["<unk>"], ((x, i) for i, x in enumerate(self.itos)))

    def __len__(self):
        return len(self.itos)

    def lookup_indices(self, tokens):
        return [self.stoi[token] for token in tokens]

    def lookup_tokens(self, indices):
        return [self.itos[index] for index in indices]
