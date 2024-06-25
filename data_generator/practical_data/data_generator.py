from xeger import Xeger
import pathlib
import argparse
import csv
import sre_parse
import string

import configparser
import re2 as re
import random
import os

from preprocess_regex import preprocess_regex, set_seed

parser = argparse.ArgumentParser()
parser.add_argument("data_name")
parser.add_argument("--augment", default=10, dest="aug_ratio", type=int, action="store", help="augmentation number")
opt = parser.parse_args()

printable_ascii = string.printable[:-5]

MAX_SEQUENCE_LENGTH = 15
EXAMPLE_NUM = 20
AUGMENTATION_RATIO = opt.aug_ratio


class PredictableException(Exception):
    pass


def get_longest_common_substring(strings):
    def get_n_grams(string: str, n: int) -> set:
        n_grams = set()
        for i in range(0, len(string) - n + 1):
            n_grams.add(string[i : i + n])
        return n_grams

    def get_all_grams(string: str) -> set:
        all_grams = set()
        for length in range(len(string) + 1):
            n_grams = get_n_grams(string, length)
            all_grams.update(n_grams)
        return all_grams

    strings_n_grams = set()
    for string in strings:
        all_grams = get_all_grams(string)
        strings_n_grams.add(frozenset(all_grams))
    intersection = frozenset.intersection(*strings_n_grams)
    longest_common_substring = max(intersection, key=len)
    return longest_common_substring


def make_pos(regex, xeger):
    pos = []
    for i in range(200):
        example_candidate = xeger.xeger(regex)
        example = ""
        for character in example_candidate:
            if character not in printable_ascii:
                character = chr(random.randint(ord(" "), ord("~")))
            example += character
        if len(example) < MAX_SEQUENCE_LENGTH and example not in pos and example:
            if bool(re.fullmatch(regex, example)):
                pos.append(example)
        if len(pos) == EXAMPLE_NUM:
            break

    # remove empty string
    pos = list(filter(None, list(pos)))

    if len(pos) != EXAMPLE_NUM:
        raise PredictableException("can not make EXAMPLE_NUM of examples")

    substitutions = dict()
    index = 0
    while True:
        lcs = get_longest_common_substring(pos)
        if len(lcs) < 2:
            break
        substitution = chr(index)
        substitutions[substitution] = lcs
        for i in range(len(pos)):
            pos[i] = pos[i].replace(lcs, substitution)
        regex_lcs = ""
        for character in lcs:
            if character in "$()*+./?[\\]^|{}":
                character = "\\" + character
            regex_lcs += character
        regex = regex.replace(regex_lcs, substitution)
        index += 1
    return regex, pos, substitutions


def make_label(regex, pos):
    # Tag preprocessing
    str_list = []
    bracket = 0
    tagIndex = 1
    for letter in regex:
        str_list.append(letter)

        if letter == "(":
            if bracket == 0:
                str_list.append("?P<t" + str(tagIndex) + ">")
                tagIndex += 1
            bracket += 1
        elif letter == ")":
            bracket -= 1
    regex = "".join(str_list)

    subregex_list = []
    bracket = 0
    for letter in regex:
        if letter == "(":
            if bracket == 0:
                subregex_list.append("")
            else:
                subregex_list[-1] = subregex_list[-1] + letter
            bracket += 1
        elif letter == ")":
            if bracket != 1:
                subregex_list[-1] = subregex_list[-1] + letter
            bracket -= 1
        else:
            subregex_list[-1] = subregex_list[-1] + letter

    SIGMA_STAR = "0"

    # generate templetes
    templete = []

    for example in pos[: EXAMPLE_NUM // 2]:
        if example != "<pad>":
            str_list = []

            dic = re.fullmatch(regex, example).groupdict()
            label_num = 1
            for i in range(1, len(dic) + 1):
                targetstring = dic["t" + str(i)]
                targetregex = re.sub("\?P\<t\d*?\>", "", subregex_list[i - 1])

                if targetregex in [".*", "\d*", "\w*", "(.)*", "(\d)*", "(\w)*", "(.*)", "(\d*)", "(\w*)", ".*?", "\d*?", "\w*?"]:
                    label = SIGMA_STAR
                else:
                    if label_num < 10:
                        label = str(label_num)
                    else:
                        label = chr(55 + label_num)
                label_num += 1
                count = len(targetstring)
                for _ in range(count):
                    str_list.append(label)
            templete.append("".join(str_list))
        else:
            templete.append("<pad>")

    for idx, pp in enumerate(pos[: EXAMPLE_NUM // 2]):
        if len(pp) != len(templete[idx]):
            raise PredictableException("lable_length error")
    return templete, subregex_list


def make_neg(regex, pos, substitutions):
    neg = []
    symbol_list = set()
    for i in pos:
        symbol_list.update(set(i))
    symbol_list.difference_update(set(substitutions.keys()))
    symbol_list = list(symbol_list)

    for i in range(0, 1000):
        # select random pos
        example = pos[random.randrange(0, len(pos))]
        count = max(int(len(example) / 5), 2)
        for _ in range(count):
            point = random.randrange(0, len(example))
            if example[point] not in ("\x00", "\x01", "\x02", "\x03", "\x04", "\x05", "\x06"):
                example = example[:point] + symbol_list[random.randrange(0, len(symbol_list))] + example[point + 1 :]

        if re.fullmatch(regex, example) is None and example not in neg:
            neg.append(example)

        if len(neg) == EXAMPLE_NUM:
            break

    if not len(neg) == EXAMPLE_NUM:
        raise PredictableException("can not make EXAMPLE_NUM of examples")

    return neg


def main():
    config = configparser.ConfigParser()
    config.read("config.ini", encoding="utf-8")

    random.seed(int(config["seed"]["practical_data"]))
    xeger = Xeger(limit=5, seed=int(config["seed"]["practical_data"]))
    os.environ["pythonhashseed"] = config["seed"]["practical_data"]
    set_seed(int(config["seed"]["practical_data"]))

    data_name = opt.data_name

    regex_file = open(f"practical_regex/{data_name}.re", "r")
    pathlib.Path("data/practical_data/org").mkdir(parents=True, exist_ok=True)
    save_file = open("data/practical_data/org/" + data_name + ".csv", "w")
    writer = csv.writer(save_file)
    regex_list = [x.strip() for x in regex_file.readlines()]
    error_idx = []
    generatable_regex_idx = set()
    unique_regex = set()
    max_len = -float("inf")

    for idx, regex in enumerate(regex_list):
        # Pre-preprocess
        if data_name == "regexlib-clean":
            pass
        elif data_name[:-3] == "practical_regexes":
            regex = regex[1:-1]
            # practical_regexes store regex as repr format thus we need to change \\ to single \
            regex = regex.replace("\\\\", "\\")
        elif data_name == "snort-clean":
            regex = regex[1 : regex.rfind("/")]
        try:
            ast = sre_parse.parse(regex)
            regex = preprocess_regex(ast, is_root=True)
        except AssertionError as e:
            error_idx.append(idx)
            continue

        except Exception as e:
            error_idx.append(idx)
            continue

        try:
            for _ in range(AUGMENTATION_RATIO):
                substituted_regex, pos, substitutions = make_pos(regex, xeger)
                neg = make_neg(substituted_regex, pos, substitutions)
                label, subregex_list = make_label(substituted_regex, pos)

                for i in range(len(subregex_list)):
                    subregex_list[i] = re.sub("\?P\<t\d*?\>", "", subregex_list[i])

                train_pos = pos[: EXAMPLE_NUM // 2]
                valid_pos = pos[EXAMPLE_NUM // 2 :]
                train_neg = neg[: EXAMPLE_NUM // 2]
                valid_neg = neg[EXAMPLE_NUM // 2 :]
                labelled_pos = label

                writer.writerow([train_pos, valid_pos, train_neg, valid_neg, labelled_pos, subregex_list])
                set2regex_goal = "".join(subregex_list)
                generatable_regex_idx.add(idx)
                unique_regex.add(set2regex_goal)

                max_len = max(max_len, len(set2regex_goal))

        except PredictableException:
            error_idx.append(idx)
            continue
        except Exception as e:
            error_idx.append(idx)
            continue

    save_file.close()
    log = {
        "data_name": data_name,
        "error count": len(error_idx),
        "total regex": len(regex_list),
        "generatable regex": len(generatable_regex_idx),
        "unique regex": len(unique_regex),
        "max regex len": max_len,
    }
    print(log)


if __name__ == "__main__":
    main()
