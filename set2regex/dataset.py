from string import printable
from ast import literal_eval

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader


# <sos>: 0
# <eos>: 1
class Vocabulary:
    def __init__(self):
        self.itos = []
        self.itos += ["<sos>", "<eos>"]
        for i in printable[:-5]:
            self.itos.append(str(i))
        self.itos += ["\x00", "\x01", "\x02", "\x03", "\x04", "\x05", "\x06", "<pad>", "<unk>"]
        self.stoi = dict((x, i) for i, x in enumerate(self.itos))

    def __len__(self):
        return len(self.itos)

    def get_idx(self, symbol):
        idx = self.stoi.get(symbol)
        return self.stoi["<unk>"] if idx is None else idx

    def get_idx_list(self, listed_example):
        return list(map(self.get_idx, listed_example))


class CustomDataset(Dataset):
    def __init__(self, file_path, string_max_len=10, regex_max_len=250):
        self.string_max_len = string_max_len
        self.regex_max_len = regex_max_len
        self.df = pd.read_csv(
            file_path,
            names=["pos", "valid_pos", "neg", "valid_neg", "label", "subregex_list"],
            converters={
                "pos": literal_eval,
                "valid_pos": literal_eval,
                "neg": literal_eval,
                "valid_neg": literal_eval,
                "label": literal_eval,
                "subregex_list": literal_eval,
            },
        )
        self.pos = self.df[self.df.columns[0]]
        self.valid_pos = self.df[self.df.columns[1]]
        self.neg = self.df[self.df.columns[2]]
        self.valid_neg = self.df[self.df.columns[3]]
        self.label = self.df[self.df.columns[4]]
        self.subregex_list = self.df[self.df.columns[5]]
        self.vocab = Vocabulary()

    def __len__(self):
        return len(self.df)

    def _translate_examples(self, examples):
        translated_examples = []
        for example in examples:
            if example == "<pad>":
                translated_example = ["<pad>"] * self.string_max_len
            else:
                # If there is <pad> at the end, it would be divided to <, p, a, d, > not intended
                translated_example = list(example) + ["<pad>"] * (self.string_max_len - len(example))
            translated_examples.append(self.vocab.get_idx_list(translated_example[: self.string_max_len]))
        return torch.tensor(translated_examples)

    def _translate_regex(self, subregex_list):
        if len(subregex_list) <= 1:
            regex = "".join(subregex_list)
        else:
            regex = "(" + ")(".join(subregex_list) + ")"
        regex = ["<sos>"] + list(regex)
        regex = regex[: self.regex_max_len - 1] + ["<eos>"]
        regex = regex + ["<pad>"] * (self.regex_max_len - len(regex))
        regex = self.vocab.get_idx_list(regex[: self.regex_max_len])
        return torch.tensor(regex)

    def __getitem__(self, idx):
        return (self._translate_examples(self.pos.iloc[idx]), self._translate_examples(self.neg.iloc[idx]), self._translate_regex(self.subregex_list.iloc[idx]))


def get_loader(file_path, batch_size, num_worker=0, shuffle=True, string_max_len=10, regex_max_len=250):
    dataset = CustomDataset(file_path, string_max_len=string_max_len, regex_max_len=regex_max_len)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_worker,
        shuffle=shuffle,
        pin_memory=True,
    )
    return loader
