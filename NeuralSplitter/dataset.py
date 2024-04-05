from string import ascii_uppercase, ascii_lowercase
from ast import literal_eval

import torch
import pandas as pd
import re2 as re
from torch.utils.data import Dataset, DataLoader


class Vocabulary:
    def __init__(self):
        self.itos = []
        for i in range(10):
            self.itos.append(str(i))
        for i in ascii_uppercase:
            self.itos.append(i)
        for i in ascii_lowercase:
            self.itos.append(i)
        self.itos += ["!", "_", "<pad>", "<unk>"]
        # String     Integer
        # 0-9        0-9
        # A-Z        10-35
        # a-z        36-61
        # !          62
        # _          63
        # <pad>      64
        # <unk>      65
        self.stoi = dict((x, i) for i, x in enumerate(self.itos))

    def __len__(self):
        return len(self.itos)

    def get_idx(self, symbol):
        idx = self.stoi.get(symbol)
        return self.stoi["!"] if idx is None else idx

    def get_idx_list(self, listed_example):
        return list(map(self.get_idx, listed_example))


class CustomDataset(Dataset):
    def __init__(self, file_path, is_test=False, max_len=10):
        self.is_test = is_test
        self.max_len = max_len
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
                translated_example = ["<pad>"] * self.max_len
            else:
                # If there is <pad> at the end, it would be divided to <, p, a, d, > not intended
                translated_example = list(example) + ["<pad>"] * (self.max_len - len(example))
            translated_examples.append(self.vocab.get_idx_list(translated_example[: self.max_len]))
        return torch.Tensor(translated_examples)

    def _pad_subregex_list(self, subregex_list):
        subregex_list = subregex_list + ["<pad>"] * (self.max_len - len(subregex_list))
        return subregex_list[: self.max_len]

    def __getitem__(self, idx):
        if self.is_test:
            return (
                self._translate_examples(self.pos.iloc[idx]),
                self._translate_examples(self.neg.iloc[idx]),
                self._pad_subregex_list(self.subregex_list.iloc[idx]),
                self._translate_examples(self.valid_pos.iloc[idx]),
                self._translate_examples(self.valid_neg.iloc[idx]),
                self._translate_examples(self.label.iloc[idx]),
            )
        else:
            return (
                self._translate_examples(self.pos.iloc[idx]),
                self._translate_examples(self.label.iloc[idx]),
                self._pad_subregex_list(self.subregex_list.iloc[idx]),
            )


def get_loader(file_path, batch_size, is_test=False, num_worker=0, shuffle=True, max_len=10):
    def collate_fn(items):
        nonlocal is_test
        batch_pos = []
        batch_neg = []
        batch_subregex = []
        batch_valid_pos = []
        batch_valid_neg = []
        batch_label = []
        if is_test:
            for pos, neg, subregex_list, valid_pos, valid_neg, label in items:
                batch_pos.append(pos)
                batch_neg.append(neg)
                batch_subregex.append(subregex_list)
                batch_valid_pos.append(valid_pos)
                batch_valid_neg.append(valid_neg)
                batch_label.append(label)
            return (
                torch.stack(batch_pos),
                torch.stack(batch_neg),
                batch_subregex,
                torch.stack(batch_valid_pos),
                torch.stack(batch_valid_neg),
                torch.stack(batch_label),
            )
        else:
            for pos, label, subregex_list in items:
                batch_pos.append(pos)
                batch_label.append(label)
                batch_subregex.append(subregex_list)
            return torch.stack(batch_pos), torch.stack(batch_label), batch_subregex

    dataset = CustomDataset(file_path, is_test, max_len=max_len)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_worker,
        shuffle=shuffle,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return loader
