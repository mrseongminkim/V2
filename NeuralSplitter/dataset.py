from string import printable
from ast import literal_eval

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class Vocabulary:
    def __init__(self):
        # 95 + 9 + 2 = 106
        self.itos = []
        for i in printable[:-5]:
            self.itos.append(str(i))
        self.itos += ["\x00", "\x01", "\x02", "\x03", "\x04", "\x05", "\x06", "<pad>", "<unk>", "<sos>", "<eos>"]
        self.stoi = dict((x, i) for i, x in enumerate(self.itos))

    def __len__(self):
        return len(self.itos)

    def get_idx(self, symbol):
        idx = self.stoi.get(symbol)
        return self.stoi["<unk>"] if idx is None else idx

    def get_idx_list(self, listed_example):
        return list(map(self.get_idx, listed_example))


class CustomDataset(Dataset):
    def __init__(self, file_path, is_test=False, max_len=10, is_set2regex=False):
        self.file_path = file_path
        self.is_set2regex = is_set2regex
        self.is_test = is_test
        self.max_len = max_len
        self.df = pd.read_csv(
            file_path,
            names=["pos", "valid_pos", "neg", "valid_neg", "label", "regex"],
            converters={
                "pos": literal_eval,
                "valid_pos": literal_eval,
                "neg": literal_eval,
                "valid_neg": literal_eval,
                "label": literal_eval,
                "regex": literal_eval,
            },
        )
        self.pos = self.df[self.df.columns[0]]
        self.valid_pos = self.df[self.df.columns[1]]
        self.neg = self.df[self.df.columns[2]]
        self.valid_neg = self.df[self.df.columns[3]]
        self.label = self.df[self.df.columns[4]]
        self.regex = self.df[self.df.columns[5]]
        self.vocab = Vocabulary()

    def __len__(self):
        return len(self.df)

    # for strings
    def _translate_examples(self, examples):
        padded_examples = []
        for example in examples:
            example = example + ["<pad>"] * (self.max_len - len(example))
            padded_examples.append(self.vocab.get_idx_list(example[: self.max_len]))
        return torch.tensor(padded_examples)

    # for regex
    def _translate_regex(self, regex):
        if "practical" in self.file_path:
            max_len = 250 + 2
        else:
            max_len = 10 + 2
        regex = ["<sos>"] + regex[: max_len - 2] + ["<eos>"]
        regex = regex + ["<pad>"] * (max_len - len(regex))
        regex = self.vocab.get_idx_list(regex[:max_len])
        return torch.tensor(regex)

    def __getitem__(self, idx):
        if self.is_test:
            return (
                self._translate_examples(self.pos.iloc[idx]),
                self._translate_examples(self.neg.iloc[idx]),
                self.subregex_list.iloc[idx],
                self._translate_examples(self.valid_pos.iloc[idx]),
                self._translate_examples(self.valid_neg.iloc[idx]),
                self._translate_examples(self.label.iloc[idx]),
            )
        elif self.is_set2regex:
            return (
                # pos, neg, label
                self._translate_examples(self.pos.iloc[idx]),
                self._translate_examples(self.neg.iloc[idx]),
                self._translate_regex(self.regex.iloc[idx]),
            )
        else:
            return (
                self._translate_examples(self.pos.iloc[idx]),
                self._translate_examples(self.label.iloc[idx]),
                self._translate_regex(self.regex.iloc[idx]),
            )


def get_loader(file_path, batch_size, is_test=False, num_worker=0, shuffle=True, max_len=10, is_set2regex=False):
    def collate_fn(items):
        nonlocal is_test
        nonlocal is_set2regex
        batch_pos = []
        batch_neg = []
        batch_regex = []
        batch_valid_pos = []
        batch_valid_neg = []
        batch_label = []
        if is_test:
            for pos, neg, regex, valid_pos, valid_neg, label in items:
                batch_pos.append(pos)
                batch_neg.append(neg)
                batch_regex.append(regex)
                batch_valid_pos.append(valid_pos)
                batch_valid_neg.append(valid_neg)
                batch_label.append(label)
            return (
                torch.stack(batch_pos),
                torch.stack(batch_neg),
                torch.stack(batch_regex),
                torch.stack(batch_valid_pos),
                torch.stack(batch_valid_neg),
                torch.stack(batch_label),
            )
        elif is_set2regex:
            for pos, neg, regex in items:
                batch_pos.append(pos)
                batch_neg.append(neg)
                batch_regex.append(regex)
            return torch.stack(batch_pos), torch.stack(batch_neg), torch.stack(batch_regex)
        else:
            for pos, label, regex in items:
                batch_pos.append(pos)
                batch_label.append(label)
                batch_regex.append(regex)
            return torch.stack(batch_pos), torch.stack(batch_label), torch.stack(batch_regex)

    dataset = CustomDataset(file_path, is_test, max_len=max_len, is_set2regex=is_set2regex)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_worker,
        shuffle=shuffle,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return loader
