from string import printable
from ast import literal_eval
from collections import defaultdict

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader


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


class RegexDataset(Dataset):
    def __init__(self, file_path, usage="train", example_max_len=10, regex_max_len=100):
        self.usage = usage
        self.example_max_len = example_max_len
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
                translated_example = ["<pad>"] * self.example_max_len
            else:
                translated_example = list(example) + ["<pad>"] * (self.example_max_len - len(example))
            translated_examples.append(self.vocab.lookup_indices(translated_example[: self.example_max_len]))
        return torch.tensor(translated_examples)

    def _translate_regex(self, regex):
        translated_regex = self.vocab.lookup_indices(["<sos>"] + list(regex)[: self.regex_max_len - 2] + ["<eos>"])
        return torch.tensor(translated_regex)

    def __getitem__(self, idx):
        if self.usage == "train":
            return self._translate_examples(self.pos.iloc[idx]), self._translate_examples(self.label.iloc[idx])
        elif self.usage == "test":
            regex = "".join(self.subregex_list.iloc[idx])
            return (
                self._translate_examples(self.pos.iloc[idx]),
                self.neg.iloc[idx],
                regex,
                self.valid_pos.iloc[idx],
                self.valid_neg.iloc[idx],
                self._translate_examples(self.label.iloc[idx]),
            )
        elif self.usage == "set2regex":
            regex = "".join(self.subregex_list.iloc[idx])
            return (
                self._translate_examples(self.pos.iloc[idx]),
                self._translate_examples(self.neg.iloc[idx]),
                self._translate_regex(regex),
            )


def get_data_loader(file_path, usage="test", example_max_len=10, regex_max_len=100, batch_size=512, num_worker=0, shuffle=True):
    dataset = RegexDataset(file_path, usage, example_max_len=example_max_len, regex_max_len=regex_max_len)
    padding_value = dataset.vocab.stoi["<pad>"]

    def collate_fn(batch):
        batch_pos = []
        batch_valid_pos = []
        batch_neg = []
        batch_valid_neg = []
        batch_label = []
        batch_regex = []
        if usage == "train":
            for pos, label in batch:
                batch_pos.append(pos)
                batch_label.append(label)
            batch_pos = torch.stack(batch_pos)
            batch_label = torch.stack(batch_label)
            return batch_pos, batch_label
        elif usage == "test":
            for pos, neg, regex, valid_pos, valid_neg, label in batch:
                batch_pos.append(pos)
                batch_neg.append(neg)
                batch_regex.append(regex)
                batch_valid_pos.append(valid_pos)
                batch_valid_neg.append(valid_neg)
                batch_label.append(label)
            batch_pos = torch.stack(batch_pos)
            batch_label = torch.stack(batch_label)
            return batch_pos, batch_neg, batch_regex, batch_valid_pos, batch_valid_neg, batch_label
        elif usage == "set2regex":
            for pos, neg, regex in batch:
                batch_pos.append(pos)
                batch_neg.append(neg)
                batch_regex.append(regex)
            batch_pos = torch.stack(batch_pos)
            batch_neg = torch.stack(batch_neg)
            batch_regex = nn.utils.rnn.pad_sequence(batch_regex, batch_first=True, padding_value=padding_value)
            return batch_pos, batch_neg, batch_regex

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_worker,
        shuffle=shuffle,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return loader


"""
debug = get_data_loader("./data/practical_data/integrated/test_snort.csv", "set2regex", 15, 1000, 1, 0, False)
for batch_pos, batch_neg, batch_regex in debug:
    print(batch_pos)
    print(batch_neg)
    print(batch_regex)
    exit()
"""
