from ast import literal_eval

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from vocabulary import Vocabulary


class RegexDataset(Dataset):
    def __init__(self, file_path, usage, regex_max_length):
        self.df = pd.read_csv(
            file_path,
            names=["train_pos", "valid_pos", "train_neg", "valid_neg", "label", "subregex_list"],
            converters={
                "train_pos": literal_eval,
                "valid_pos": literal_eval,
                "train_neg": literal_eval,
                "valid_neg": literal_eval,
                "label": literal_eval,
                "subregex_list": literal_eval,
            },
        )
        self.vocab = Vocabulary()
        self.usage = usage
        self.regex_max_length = regex_max_length
        self.train_pos = self.df[self.df.columns[0]]
        self.valid_pos = self.df[self.df.columns[1]]
        self.train_neg = self.df[self.df.columns[2]]
        self.valid_neg = self.df[self.df.columns[3]]
        self.label = self.df[self.df.columns[4]]
        self.subregex_list = self.df[self.df.columns[5]]

    def __len__(self):
        return len(self.df)

    def tokenize(self, list_of_string):
        # Tokenize string
        if isinstance(list_of_string, str):
            return torch.tensor(self.vocab.lookup_indices(["<sos>"] + list(list_of_string)[: self.regex_max_length] + ["<eos>"]))
        else:
            return [torch.tensor(self.vocab.lookup_indices(list(string))) for string in list_of_string]

    def __getitem__(self, index):
        train_pos = self.train_pos[index]
        valid_pos = self.valid_pos[index]
        train_neg = self.train_neg[index]
        valid_neg = self.valid_neg[index]
        label = self.label[index]
        regex = "".join(self.subregex_list[index])
        if self.usage == "train":
            return {
                "train_pos_ids": self.tokenize(train_pos),
                "label_ids": self.tokenize(label),
            }
        elif self.usage == "synthesis":
            return {
                "train_pos": train_pos,
                "train_pos_ids": self.tokenize(train_pos),
                "valid_pos": valid_pos,
                "train_neg": train_neg,
                "train_neg_ids": self.tokenize(train_neg),
                "valid_neg": valid_neg,
                "label": label,
            }
        elif self.usage == "set2regex":
            return {
                "train_pos_ids": self.tokenize(train_pos),
                "train_neg_ids": self.tokenize(train_neg),
                "regex_ids": self.tokenize(regex),
            }


def get_data_loader(file_path, usage, pad_index, regex_max_length, batch_size, num_worker, shuffle):
    def collate_fn(batch):
        if usage == "train":
            batch_train_pos_ids = []
            batch_label_ids = []
            for data in batch:
                batch_train_pos_ids.extend(data["train_pos_ids"])
                batch_train_pos_ids.extend(data["label_ids"])
            print(batch_train_pos_ids)
            # batch_train_pos_ids = nn.utils.rnn.pad_sequence(batch_train_pos_ids, padding_value=pad_index)
            # print(batch_train_pos_ids)
        elif usage == "synthesis":
            pass
        elif usage == "set2regex":
            batch_train_pos_ids = []
            batch_train_neg_ids = []
            batch_regex_ids = []
            for data in batch:
                batch_train_pos_ids.extend(data["train_pos_ids"])
                batch_train_neg_ids.extend(data["train_neg_ids"])
                batch_regex_ids.append(data["regex_ids"])
            batch_train_ids = batch_train_pos_ids + batch_train_neg_ids
            batch_train_ids = nn.utils.rnn.pad_sequence(batch_train_ids, padding_value=pad_index)
            batch_train_pos_ids = batch_train_ids[:, : batch_train_ids.size(1) // 2].detach()
            batch_train_neg_ids = batch_train_ids[:, batch_train_ids.size(1) // 2 :].detach()
            batch_regex_ids = nn.utils.rnn.pad_sequence(batch_regex_ids, padding_value=pad_index)
            batch = {
                "pos": batch_train_pos_ids,
                "neg": batch_train_neg_ids,
                "regex": batch_regex_ids,
            }
            return batch

    dataset = RegexDataset(file_path, usage, regex_max_length)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_worker,
        shuffle=shuffle,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return loader
