from xeger import Xeger
import pathlib
import argparse
import os, sys

import configparser
import re2 as re
import random

dequantifier = "(\\\.|\\.|\\\\x..)"
dequantifier2 = "(\(.*?\)|\[[^]]*\])"
dequantifier5 = "\\\\d|\\\\D\\\\|\\\\w|\\\\W|\\\\s|\\\\S|(?<!\\\\)\."

quantifier = "(\*|\+|\?|\{\d+\,\d*\}|\{\d+\})\??"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--augment", default=10, dest="aug_ratio", type=int, action="store", help="augmentation number"
)
opt = parser.parse_args()

MAX_SEQUENCE_LENGTH = 15
EXAMPLE_NUM = 20
AUGMENTATION_RATIO = opt.aug_ratio

# # Description
# Preprocess the practical data from submodule.(snort, regexlib, Polyglot Corpus)
# Appear in data/practical/org
# Each pos, neg strings have a maximum length of 'MAX_SEQUENCE_LENGTH'.
# Each data have 'EXAMPLE_NUM' of pos, neg strings.
# Original datasets in submodule are multiply by 'AUGMENTATION_RATIO'


class PredictableException(Exception):
    pass


def make_pos(regex, xeger):
    pos = []

    for i in range(200):
        example_candidate = xeger.xeger(regex)
        if len(example_candidate) < MAX_SEQUENCE_LENGTH and example_candidate not in pos:
            pos.append(example_candidate)
        if len(pos) == EXAMPLE_NUM:
            break

    # remove empty string
    pos = list(filter(None, list(pos)))

    if len(pos) != EXAMPLE_NUM:
        #operator 없는 것들이 있어서 충분히 가능하다.
        raise PredictableException("can not make EXAMPLE_NUM of examples")

    return pos


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

    for example in pos:
        if example != "<pad>":
            str_list = []

            example = re.sub("\x0b", "`", example)

            xx = re.fullmatch(regex, example)
            if xx is None:

                example = re.sub("`", "\t", example)

            dic = re.fullmatch(regex, example).groupdict()
            label_num = 1
            for i in range(1, len(dic) + 1):

                targetstring = dic["t" + str(i)]
                targetregex = re.sub("\?P\<t\d*?\>", "", subregex_list[i - 1])

                if re.fullmatch(r"(\.|\[.*(\\d|\\D|\\w|\\W|\\S).*\])\*", targetregex):
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

    for idx, pp in enumerate(pos):
        if len(pp) != len(templete[idx]):
            raise PredictableException("lable_length error")
    return templete


def make_neg(regex, pos):
    neg = []
    symbol_list = []

    for i in pos:
        symbol_candidates = re.findall("(?!\x0b|\\\\|\\|').", i)
        for symbol in symbol_candidates:
            if symbol not in symbol_list:
                symbol_list.append(symbol)

    for i in range(0, 1000):
        # select random pos
        example = pos[random.randrange(0, len(pos))]

        count = max(int(len(example) / 5), 2)
        for j in range(count):
            point = random.randrange(0, len(example))
            if example[point] != "'" and example[point] != r"\\":
                example = (
                    example[:point]
                    + symbol_list[random.randrange(0, len(symbol_list))]
                    + example[point + 1 :]
                )

        if re.fullmatch(regex, example) is None and example not in neg:
            neg.append(example)

        if len(neg) == EXAMPLE_NUM:
            break

    if not len(neg) == EXAMPLE_NUM:
        raise PredictableException("can not make EXAMPLE_NUM of examples")

    return neg


def remove_anchor(regex):
    # \x5c = \

    # Negative lookbehind - Not supported
    # \^: ^ literal
    # [^: negation
    # 위 두 가지를 제외하고 ^를 모두 지운다.
    regex = re.sub(r"(?<!\x5c|\[)\^", "", regex)

    # Negative lookbehind - Not supported
    # \$: $ literal
    # literal $을 제외하고 모두 지운다.
    regex = re.sub(r"(?<!\x5c)\$", "", regex)

    # \A: at beginning of text
    # \Z: at end of text
    # \z: at end of text
    # \a: beep sound 😀
    # "\\Apdf\\z" 이런 경우에는 오류가 발생하지 않을까?
    regex = re.sub(r"\x5cA|\x5cZ|\x5cz", "", regex)

    return regex


def remove_redundant_quantifier(regex):
    # repetition 뒤에 +가 오는 경우를 제거
    # ab{2,3}* 같은 경우가 있으니 }*도?
    # "[Q\\u0100\\u0200]{1,3}?" 같은 경우도 있다.
    # "\\(\\p{ASCII}*\\)"

    #{n,m}?; prefer fewer
    #{n,}?: prefer fewer
    #{n}?: exactly n
    #{n,m}+: possessive
    #{n,}+: possessive
    #{n}+: exactly n

    #함부로 지울 수 없는 이유
    #"\\p{XDigit}*"
    #이 경우에 {}는 literal이다.
    #그니까 앞에 {%d, } 이거 체크하는 것이 필요할 것 같다.
    #regex = re.sub("}\+", "}", regex)
    #print(regex)
    while True:
        # subn -> (new_string, number_of_subs_made)
        #print(re.match("(\[[^]]*\]|\\\.|\\.|\(.*?\))" + "((\+|\{\d+\,\d*\}|\{\d+\})\??|\*\?)", regex))
        #print(re.fullmatch("(\[[^]]*\]|\\\.|\\.|\(.*?\))" + "((\+|\{\d+\,\d*\}|\{\d+\})\??|\*\?)", regex))
        #이거 원본 뭐였지
        #repretition이 있으면 그 대상(\1)을 *로 바꿔준다.



        # Cover
        # {n,m} with ?, +, *
        # {n,} with ?, +, *
        # {n} with ?, +, *
        # + -> *; idk why
        regex, a = re.subn(
            "(\[[^]]*\]|\\\.|\\.|\(.*?\))" + "(\+|\{\d+(\,\d*)?\}[\?\*\+]*)", r"\1*", regex
        )
        #print("aaaaaa:", regex)
        #print(a)
        #exit()
        regex, b = re.subn("(\[[^]]*\]|\\\.|\\.|\(.*?\))" + "(\?\?)", r"\1?", regex)
        #print("bbbbbb:", regex)
        regex, c = re.subn(
            r"(\\x[0-9A-Fa-f][0-9A-Fa-f]|@)" + "((\+|\?|\{\d+\,\d*\}|\{\d+\})\??|\*\?)",
            r"\1*",
            regex,
        )
        if a + b + c == 0:
            break

    # remove back reference
    regex = re.sub(r"\\\d", r"", regex)

    return regex


def preprocess_parenthesis_flag(regex):
    # unicode
    if re.search(r"\x5cu", regex) is not None:
        raise PredictableException("There is a unicode problem")

    # lookahead
    if re.search(r"\(\?=|\(\?<=|\(\?!|\(\?<!", regex) is not None:
        raise PredictableException("There is a lookahead problem")

    # non capturing
    regex = re.sub(r"\(\?:", "(", regex)

    # named group
    regex = re.sub(r"\(\?P<.*?>", "(", regex)

    regex = re.sub(r"\(\?<.*?>", "(", regex)
    # non-backtracking group
    regex = re.sub(r"\(\?>", "(", regex)

    regex = re.sub(r"\\b", r"", regex)
    regex = re.sub(r"\\B", r"", regex)
    regex = re.sub(r"\\k", "", regex)

    regex = re.sub(r"\\\[", r"!", regex)
    regex = re.sub(r"\\\]", r"!", regex)
    regex = re.sub(r"\\\(", r"!", regex)
    regex = re.sub(r"\\\)", r"!", regex)

    # remove not operator
    regex = re.sub("(?<=\[)\^([^]]*?)(?=\])", r"\1", regex)
    regex = re.sub("\\\s", "`", regex)

    return regex


def preprocess_replace(regex):
    # control_ascii
    regex = re.sub(r"\\x([\d][0-9A-Fa-f])", r"!", regex)

    # space_character
    regex = re.sub(r"\\r", r"!", regex)
    regex = re.sub(r"\\n", r"!", regex)
    regex = re.sub(r"\\t", r"!", regex)
    regex = re.sub(r" ", r"!", regex)

    regex = re.sub(r"\\\\", r"!", regex)
    regex = re.sub(r"\\\'", r"!", regex)
    regex = re.sub(r"\\", r"!", regex)

    regex = re.sub(r"\\x5(c|C)", r"!", regex)

    regex = re.sub("[^\w]", r"!", regex)

    return regex


def get_captured_regex(regex):
    matchObj_iter = re.finditer(
        dequantifier
        + quantifier
        + "|"
        + dequantifier2
        + "("
        + quantifier
        + ")?"
        + "|"
        + dequantifier5,
        regex,
    )

    split_point = [0]
    indicate = 1

    regex = "(" + regex
    for matchObj in matchObj_iter:
        start, end = matchObj.start(), matchObj.end()
        regex = regex[: start + indicate] + ")" + regex[start + indicate :]
        indicate += 1
        regex = regex[: start + indicate] + "(" + regex[start + indicate :]
        indicate += 1
        regex = regex[: end + indicate] + ")" + regex[end + indicate :]
        indicate += 1
        regex = regex[: end + indicate] + "(" + regex[end + indicate :]
        indicate += 1
        split_point.append(start)
        split_point.append(end)
    regex = regex + ")"

    regex = re.sub("\(\)", "", regex)

    return regex


def special_characterize(regex):
    regex = re.sub("(\\\\)?(\@|\#|\~|\`|\%|\&|\<|\>|\,|\=|'|\"| |\:|\;)", "!", regex)
    regex = re.sub("(\\\\)(\+|\*|\^|\?|\-)", "!", regex)

    regex = re.sub("(\\\\)\.", "!", regex)
    regex = re.sub(r"\x5cr|\x5cn|\x5ct", "!", regex)
    regex = re.sub("\\\\x..", "!", regex)
    return regex


def replace_constant_string(regex):
    mapping_table = {}

    # make subregex list
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

    # replace 1st strings
    for idx, subregex in enumerate(subregex_list):
        if (
            re.search(
                dequantifier
                + quantifier
                + "|"
                + dequantifier2
                + "("
                + quantifier
                + ")?"
                + "|"
                + dequantifier5,
                subregex,
            )
            is None
        ):
            if subregex in mapping_table.values():
                for alphabet, string in mapping_table.items():
                    if string == subregex:
                        ch = alphabet
            else:
                if len(mapping_table) < 26:
                    ch = chr(len(mapping_table) + 65)
                else:
                    raise PredictableException("too many constant string")
                mapping_table[ch] = subregex
            regex = re.sub(repr(subregex), ch, regex, 1)
            subregex_list[idx] = ch

        if re.fullmatch("\(.*\)", subregex_list[idx]) is None:
            subregex_list[idx] = "(" + subregex_list[idx] + ")"

    regex = "".join(subregex_list)

    string_pattern = "(?<!\\\\)[^\\\(\)\*\+\|\^\[\]\!\?]{2,}"
    while re.search(string_pattern, regex) is not None:
        tmp = re.search(string_pattern, regex).group()
        if tmp in mapping_table.values():
            for alphabet, string in mapping_table.items():
                if string == tmp:
                    ch = alphabet
        else:
            if len(mapping_table) < 26:
                ch = chr(len(mapping_table) + 65)
            else:
                raise PredictableException("too many constant string")
            mapping_table[ch] = tmp

        regex = re.sub(string_pattern, ch, regex, 1)

    regex = re.sub("\-", "!", regex)

    string_pattern = "(?<!\\\\)[a-z]"
    while re.search(string_pattern, regex) is not None:
        tmp = re.search(string_pattern, regex).group()
        if tmp in mapping_table.values():
            for alphabet, string in mapping_table.items():
                if string == tmp:
                    ch = alphabet
        else:
            if len(mapping_table) < 26:
                ch = chr(len(mapping_table) + 65)
            else:
                raise PredictableException("too many constant string")

            mapping_table[ch] = tmp

        regex = re.sub(string_pattern, ch, regex, 1)

    return regex, mapping_table


def main():
    config = configparser.ConfigParser()
    config.read("config.ini", encoding="utf-8")

    random.seed(int(config["seed"]["practical_data"]))

    xeger = Xeger(limit=5)
    xeger.seed(int(config["seed"]["practical_data"]))

    data_pathes = [
        "practical_regex/snort-clean.re",
        "practical_regex/regexlib-clean.re",
        "practical_regex/practical_regexes.re"
    ]

    for data_path in data_pathes:
        regex_file = open(data_path, "r")
        data_name = re.search("[^/]*?(?=\.r|\.j)", data_path).group()
        pathlib.Path("data/practical_data/org").mkdir(parents=True, exist_ok=True)
        save_file = open("data/practical_data/org/" + data_name + ".csv", "w")

        print("Preprocessing " + data_name + "...")

        regex_list = [x.strip() for x in regex_file.readlines()]

        error_idx = []

        for idx, regex in enumerate(regex_list):
            if data_name == "regexlib-clean":
                regex = re.sub(r"\\\\", "\x5c", regex)
            elif data_name == "practical_regexes":
                #양 끝이 "로 감싸져있기에 지워준다.
                regex = regex[1:-1]
                # \\\\ -> !
                regex = re.sub(r"\\\\\\\\", "!", regex)
                # \\ -> \
                # \x5c = \
                regex = re.sub(r"\\\\", "\x5c", regex)
                # \x00 = NULL
                regex = re.sub(r"\x00", "", regex)
            try:
                regex = regex_list[9]
                # preprocess
                print("origin:", regex)
                regex = remove_anchor(regex)
                print("anchor:", regex)
                regex = remove_redundant_quantifier(regex)
                print("qunati:", regex)
                #exit()
                regex = preprocess_parenthesis_flag(regex)
                print("parent:", regex)
                regex = special_characterize(regex)
                print("specia:", regex)
                regex = get_captured_regex(regex)
                print("captur:", regex)
                regex, mapping_table = replace_constant_string(regex)
                print("consta:", regex)
                #exit()

                if re.search(r"(?<!\x5c)\[[^\[\]]*[()][^\[\]]*\](?!\x5c)", regex) is not None:
                    raise PredictableException("overlapped backet")

            except Exception as e:
                # if not isinstance(e, PredictableException) and not isinstance(e, re.error):
                error_idx.append(idx)
                continue

            try:
                for _ in range(AUGMENTATION_RATIO):

                    # generate pos, neg, label
                    pos = make_pos(regex, xeger)
                    neg = make_neg(regex, pos)
                    label = make_label(regex, pos)

                    # replace unrecognized symbol
                    pos = list(map(lambda y: preprocess_replace(repr(y)[1:-1]), pos))
                    neg = list(map(lambda y: preprocess_replace(repr(y)[1:-1]), neg))

                    total = pos + neg + label

                    res = ""
                    for ele in total:
                        res = res + str(ele) + ", "
                    res = res + str(regex)

                    save_file.write(res + "\n")
                exit()

            except Exception as e:
                # if not isinstance(e, PredictableException) and not isinstance(e, re.error):
                error_idx.append(idx)
                exit()
                continue

            if idx % 1000 == 0:
                print(idx)

        print("error count :", len(error_idx))
        print("total len:", len(regex_list))

if __name__ == "__main__":
    main()
