from ast import literal_eval
import pathlib
import csv

import pandas as pd
import numpy as np

import NeuralSplitter.dataset as dataset
from split import split
from synthesis import convert_indices_to_strings

debug = "./data/practical_data/integrated/test_snort.csv"
train = "./data/practical_data/integrated/train.csv"
valid = "./data/practical_data/integrated/valid.csv"

pathlib.Path("data/practical_data/set2regex").mkdir(parents=True, exist_ok=True)
set2regex_train = open("data/practical_data/set2regex/train.csv", "w")
train_writer = csv.writer(set2regex_train)
set2regex_valid = open("data/practical_data/set2regex/valid.csv", "w")
valid_writer = csv.writer(set2regex_valid)


def preprocess(file, writer):
    data = dataset.get_data_loader(file, usage="set2regex_preprocess", example_max_len=15, regex_max_len=100, batch_size=1, num_worker=0, shuffle=False)
    for pos, neg, subregex_list, label in data:
        pos_str = convert_indices_to_strings(pos)
        writer.writerow([pos_str, neg, subregex_list])

        pos = pos.unsqueeze(0)
        label = label.unsqueeze(0)
        splited_pos, sigma_lst = split(pos, np.array(label, dtype="object").T, need_bf_escape=False)

        splited_pos = splited_pos[0]
        sigma_lst = sigma_lst[0]

        split_size = len(splited_pos[0])
        split_set = []
        for sub_id in range(split_size):
            pos = []
            for set_idx in range(len(splited_pos)):
                pos.append(splited_pos[set_idx][sub_id])
            split_set.append([set(pos), set(neg)])

        for sub_id in range(split_size):
            split_set[sub_id][1] -= split_set[sub_id][0]

        for i in range(len(split_set)):
            sub_pos = list(split_set[i][0])
            sub_pos += ["<pad>"] * (10 - len(sub_pos))
            sub_neg = list(split_set[i][1])
            sub_neg += ["<pad>"] * (10 - len(sub_neg))
            sub_regex = [subregex_list[i]]
            writer.writerow([sub_pos, sub_neg, sub_regex])


preprocess(train, train_writer)
preprocess(valid, valid_writer)
