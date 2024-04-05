import pathlib
import argparse
import configparser
import random
import csv

import re2 as re
from xeger import Xeger

from parsetree import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "--alphabet_size",
    action="store",
    dest="alphabet_size",
    help="Specify the size of the alphabet used in the regular expression.",
    type=int,
    default=5,
)
parser.add_argument(
    "--is_train",
    action="store_true",
    dest="is_train",
    help="Specify whether the data is intended for training or testing purposes.",
    default=False,
)
parser.add_argument(
    "--number",
    action="store",
    dest="number",
    type=int,
    help="Specify the number of data samples to be processed.",
    default=10_000,
)
opt = parser.parse_args()

MAX_SEQUENCE_LENGTH = 10
EXAMPLE_NUM = 20
TRAIN_SEED = 10000
TEST_SEED = 20000
MAX_DEPTH = 4
SIGMA_STAR = str(KleenStar(Or(*[Character(str(x)) for x in range(opt.alphabet_size)])))
SIGMA_STAR_LABEL = "0"


def generate_random_regex(alphabet_size=5):
    regex = REGEX()
    for _ in range(MAX_DEPTH):
        regex.make_child(alphabet_size=alphabet_size)
    regex.spreadRand(alphabet_size=alphabet_size)
    return regex


def get_concise_regex():
    while True:
        regex = generate_random_regex(opt.alphabet_size)
        if (
            regex.starnormalform()
            or regex.redundant_concat1()
            or regex.redundant_concat2(opt.alphabet_size)
            or regex.KCK(opt.alphabet_size)
            or regex.KCQ(opt.alphabet_size)
            or regex.QC()
            or regex.OQ()
            or regex.orinclusive(opt.alphabet_size)
            or regex.prefix()
            or regex.sigmastar(opt.alphabet_size)
        ):
            continue
        else:
            break
    return regex.repr_labeled()


def get_pos(regex, xeger):
    pos = set()
    for _ in range(200):
        example = xeger.xeger(regex)
        if 0 < len(example) <= MAX_SEQUENCE_LENGTH:
            pos.add(example)
        if len(pos) == EXAMPLE_NUM:
            break
    return list(pos)


def get_neg(regex):
    neg = set()
    for _ in range(1_000):
        random_string = ""
        for _ in range(random.randrange(1, MAX_SEQUENCE_LENGTH + 1)):
            random_string += str(random.randrange(0, opt.alphabet_size))
        if not bool(re.fullmatch(regex, random_string)):
            neg.add(random_string)
        if len(neg) == EXAMPLE_NUM:
            break
    return list(neg)


def tag_and_split(regex):
    tagged_regex = ""
    subregex = ""
    subregex_list = []
    parenthesis = 0
    tag_index = 1
    for i in regex:
        subregex += i
        if i == "(":
            if parenthesis == 0:
                tag = f"?P<t{tag_index}>"
                subregex += tag
                tag_index += 1
            parenthesis += 1
        elif i == ")":
            parenthesis -= 1
            if parenthesis == 0:
                subregex_list.append(subregex[len(tag) + 1 : -1])
                tagged_regex += subregex
                subregex = ""
    return tagged_regex, subregex_list


def label_pos(tagged_regex, pos, subregex_list):
    labelled_pos = []
    for example in pos:
        group_dict = re.fullmatch(tagged_regex, example).groupdict()
        labelled_example = ""
        label_index = 1
        for i in range(len(group_dict)):
            target_string = group_dict[f"t{i + 1}"]
            target_regex = subregex_list[i]
            if target_regex == SIGMA_STAR:
                label = SIGMA_STAR_LABEL
            else:
                if label_index < 10:
                    label = str(label_index)
                else:
                    label = chr(55 + label_index)  # 'A', 'B', and so on
            label_index += 1
            labelled_example += label * len(target_string)
        labelled_pos.append(labelled_example)
    return labelled_pos


def generate_and_store_data():
    config = configparser.ConfigParser()
    config.read("config.ini", encoding="utf-8")

    xeger = Xeger(limit=5)

    pathlib.Path(f"./data/random_data/size_{opt.alphabet_size}").mkdir(parents=True, exist_ok=True)

    if opt.is_train:
        save_file = open(f"./data/random_data/size_{opt.alphabet_size}/train.csv", "w")
        random.seed(int(config["seed"]["random_data"]) + int(opt.alphabet_size) + TRAIN_SEED)
        xeger.seed(int(config["seed"]["random_data"]) + int(opt.alphabet_size) + TRAIN_SEED)
    else:
        save_file = open(f"./data/random_data/size_{opt.alphabet_size}/test.csv", "w")
        random.seed(int(config["seed"]["random_data"]) + int(opt.alphabet_size) + TEST_SEED)
        xeger.seed(int(config["seed"]["random_data"]) + int(opt.alphabet_size) + TEST_SEED)
    writer = csv.writer(save_file)

    data_num = 0
    while data_num < opt.number:
        regex = get_concise_regex()
        pos = get_pos(regex, xeger)
        if len(pos) != EXAMPLE_NUM:
            continue
        neg = get_neg(regex)
        if len(neg) != EXAMPLE_NUM:
            continue
        tagged_regex, subregex_list = tag_and_split(regex)
        train_pos = pos[: EXAMPLE_NUM // 2]
        valid_pos = pos[EXAMPLE_NUM // 2 :]
        train_neg = neg[: EXAMPLE_NUM // 2]
        valid_neg = neg[EXAMPLE_NUM // 2 :]
        labelled_pos = label_pos(tagged_regex, train_pos, subregex_list)
        writer.writerow([train_pos, valid_pos, train_neg, valid_neg, labelled_pos, subregex_list])
        data_num += 1
    save_file.close()


def main():
    generate_and_store_data()


if __name__ == "__main__":
    main()
