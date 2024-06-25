import random
import configparser
import pathlib
from ast import literal_eval

import pandas as pd

config = configparser.ConfigParser()
config.read("config.ini", encoding="utf-8")
random.seed(int(config["seed"]["integrate_data"]))

practical_data_list = []
for i in range(1, 55):
    practical_data_list.append(f"./data/practical_data/org/practical_regexes_{i:02}.csv")

with open("./data/practical_data/org/practical_regexes.csv", "w") as f:
    for path in practical_data_list:
        file = open(path, "r")
        f.writelines(file.readlines())

data_list = [
    "./data/practical_data/org/snort-clean.csv",
    "./data/practical_data/org/regexlib-clean.csv",
    "./data/practical_data/org/practical_regexes.csv",
]


dfs = []
for path in data_list:
    df = pd.read_csv(
        path,
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
    dfs.append(df)
snort = dfs[0]
lib = dfs[1]
prac = dfs[2]

snort = snort[~snort["subregex_list"].isin(prac["subregex_list"])]
lib = lib[~lib["subregex_list"].isin(prac["subregex_list"])]
prac = prac.sample(frac=1, random_state=int(config["seed"]["integrate_data"])).reset_index(drop=True)

split_index = int(len(prac) * 0.9)
train = prac.iloc[:split_index].reset_index(drop=True)
valid = prac.iloc[split_index:].reset_index(drop=True)

pathlib.Path("./data/practical_data/integrated").mkdir(parents=True, exist_ok=True)
snort.to_csv("data/practical_data/integrated/test_snort.csv", header=False, index=False)
lib.to_csv("data/practical_data/integrated/test_regexlib.csv", header=False, index=False)
train.to_csv("data/practical_data/integrated/train.csv", header=False, index=False)
valid.to_csv("data/practical_data/integrated/valid.csv", header=False, index=False)
