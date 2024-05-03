from xeger import Xeger
import pathlib
import argparse
import os, sys
import csv
import string

import configparser
import re2 as re
import random

printable = string.printable[:-5]

dequantifier = "(\\\.|\\.|\\\\x..)"
dequantifier2 = "(\(.*?\)|\[[^]]*\])"
dequantifier5 = "\\\\d|\\\\D\\\\|\\\\w|\\\\W|\\\\s|\\\\S|(?<!\\\\)\."

quantifier = "(\*|\+|\?|\{\d+\,\d*\}|\{\d+\})\??"

parser = argparse.ArgumentParser()
parser.add_argument("data_name")
parser.add_argument("--augment", default=10, dest="aug_ratio", type=int, action="store", help="augmentation number")
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
        if len(example_candidate) < MAX_SEQUENCE_LENGTH and example_candidate not in pos:
            example = ""
            for i in range(len(example_candidate)):
                if example_candidate[i] in printable:
                    example += example_candidate[i]
                else:
                    c = random.choice(printable)
                    example += c
            pos.append(example)
        if len(pos) == EXAMPLE_NUM:
            break

    # remove empty string
    pos = list(filter(None, list(pos)))

    if len(pos) != EXAMPLE_NUM:
        # operator 없는 것들이 있어서 충분히 가능하다.
        raise PredictableException("can not make EXAMPLE_NUM of examples")

    # print("pos:", pos)

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
        # lcs 처리
        # print("lcs:", repr(lcs))

        escaped_lcs = ""
        for i in lcs:
            if i in (".", "+", "*", "?", "^", "$", "(", ")", "[", "]", "{", "}", "|", "/"):
                i = "\\" + i
            escaped_lcs += i
        # print("escaped_lcs:", escaped_lcs)
        regex = regex.replace(escaped_lcs, substitution)
        # print("regex:", regex)
        index += 1
    # print("regex:", repr(regex))
    # print("pos:", pos)
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

    # print("tagged regex:", regex)

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

            # print("final regex:", repr(regex))
            # print("final_regex:", regex)
            # print("example:", repr(example))

            dic = re.fullmatch(regex, example).groupdict()
            label_num = 1
            for i in range(1, len(dic) + 1):

                targetstring = dic["t" + str(i)]
                targetregex = re.sub("\?P\<t\d*?\>", "", subregex_list[i - 1])
                if re.fullmatch(r"\(*?(.|\\d|\\D|\\w|\\W|\\S)\)*?\*\)*", targetregex):
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

        # print("here")
        # print("regex:", repr(regex))
        # print("example:", repr(example))
        if re.fullmatch(regex, example) is None and example not in neg:
            neg.append(example)
        # print("not here")

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

    # {n,m}?; prefer fewer
    # {n,}?: prefer fewer
    # {n}?: exactly n
    # {n,m}+: possessive
    # {n,}+: possessive
    # {n}+: exactly n

    # 함부로 지울 수 없는 이유
    # "\\p{XDigit}*"
    # 이 경우에 {}는 literal이다.
    # 그니까 앞에 {%d, } 이거 체크하는 것이 필요할 것 같다.
    # regex = re.sub("}\+", "}", regex)
    # print(regex)
    while True:
        # subn -> (new_string, number_of_subs_made)
        # print(re.match("(\[[^]]*\]|\\\.|\\.|\(.*?\))" + "((\+|\{\d+\,\d*\}|\{\d+\})\??|\*\?)", regex))
        # print(re.fullmatch("(\[[^]]*\]|\\\.|\\.|\(.*?\))" + "((\+|\{\d+\,\d*\}|\{\d+\})\??|\*\?)", regex))
        # 이거 원본 뭐였지
        # repretition이 있으면 그 대상(\1)을 *로 바꿔준다.

        # Cover
        # {n,m} with ?, +, *
        # {n,} with ?, +, *
        # {n} with ?, +, *
        # + -> *; idk why
        regex, a = re.subn("(\[[^]]*\]|\\\.|\\.|\(.*?\))" + "(\+|\{\d+(\,\d*)?\}[\?\*\+]*)", r"\1*", regex)
        regex, b = re.subn("(\[[^]]*\]|\\\.|\\.|\(.*?\))" + "(\?\?)", r"\1?", regex)
        regex, c = re.subn(
            r"(\\x[0-9A-Fa-f][0-9A-Fa-f]|@)" + "((\+|\?|\{\d+\,\d*\}|\{\d+\})\??|\*\?)",
            r"\1*",
            regex,
        )
        if a + b + c == 0:
            break

    # remove back reference
    # regex = re.sub(r"\\\d", r"", regex)

    if re.search("(\+|\{\d+(\,\d*)?\}[\?\*\+]*)", regex):
        raise PredictableException("STH is not right")

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
    # print("regex:", regex)
    matchObj_iter = re.finditer(
        dequantifier + quantifier + "|" + dequantifier2 + "(" + quantifier + ")?" + "|" + dequantifier5,
        regex,
    )

    # split_point = [0]
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
        # split_point.append(start)
        # split_point.append(end)
    regex = regex + ")"

    regex = re.sub("\(\)", "", regex)

    # print("regex-modified:", regex)
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
                dequantifier + quantifier + "|" + dequantifier2 + "(" + quantifier + ")?" + "|" + dequantifier5,
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


def replace_hex(match):
    hex_value = match.group(1)
    char = chr(int(hex_value, 16))
    # handle special characters except \; python will automatically convert \ to \\
    if char in (".", "+", "*", "?", "^", "$", "(", ")", "[", "]", "{", "}", "|"):
        char = "\\" + char
    return char


def main():
    max_len = -1

    config = configparser.ConfigParser()
    config.read("config.ini", encoding="utf-8")

    random.seed(int(config["seed"]["practical_data"]))

    xeger = Xeger(limit=5)
    xeger.seed(int(config["seed"]["practical_data"]))

    data_name = opt.data_name

    """
    data_pathes = [
        "practical_regex/snort-clean.re",
        "practical_regex/regexlib-clean.re",
        "practical_regex/practical_regexes.re"
    ]
    """
    regex_file = open(f"practical_regex/{data_name}.re", "r")
    # data_name = re.search("[^/]*?(?=\.r|\.j)", data_path).group()
    pathlib.Path("data/practical_data/org").mkdir(parents=True, exist_ok=True)
    save_file = open("data/practical_data/org/" + data_name + ".csv", "w")
    writer = csv.writer(save_file)
    # print("Preprocessing " + data_name + "...")
    regex_list = [x.strip() for x in regex_file.readlines()]
    error_idx = []
    for idx, regex in enumerate(regex_list):
        # if idx != 87:
        #    continue
        # regex = regex_list[10]
        # idx = 9221
        # regex = regex_list[idx]
        # print(regex)
        # exit()
        if data_name == "regexlib-clean":
            pass
            # regex = re.sub(r"\\\\", "\x5c", regex)
        elif data_name[:-3] == "practical_regexes":
            # 양 끝이 "로 감싸져있기에 지워준다.
            regex = regex[1:-1]
            # Xeger takes too long on this ones, and probably not usable anyway
            if regex == "(?:a(?:b(?:c(?:d(?:e(?:f(?:g(?:h(?:i(?:j(?:k(?:l(?:FooBar){0,2}){0,2}){0,2}){0,2}){0,2}){0,2}){0,2}){0,2}){0,2}){0,2}){0,2}){0,2}){0,2}":
                error_idx.append(idx)
                continue
            elif regex == "(?:a(?:b(?:c(?:d(?:e(?:f(?:g(?:h(?:i(?:j(?:k(?:l(?:m(?:n(?:o(?:p(?:q(?:r(?:s(?:t(?:u(?:v(?:w(?:x(?:y(?:z(?:FooBar)+)+)+)+)+)+)+)+)+)+)+)+)+)+)+)+)+)+)+)+)+)+)+)+)+)+)":
                error_idx.append(idx)
                continue
            elif regex == "(?:a(?:b(?:c(?:d(?:e(?:f(?:g(?:h(?:i(?:j(?:k(?:l(?:FooBar){2}){2}){2}){2}){2}){2}){2}){2}){2}){2}){2}){2}){2}":
                error_idx.append(idx)
                continue
            pass
            # \\\\ -> !
            # regex = re.sub(r"\\\\\\\\", "!", regex)
            # \\ -> \
            # \x5c = \
            # regex = re.sub(r"\\\\", "\x5c", regex)
            # \x00 = NULL
            # regex = re.sub(r"\x00", "", regex)
        elif data_name == "snort-clean":
            regex = regex[1 : regex.rfind("/")]
        try:
            # print("original:", regex)
            # If repl is a function, it is called for every non-overlapping occurrence of pattern.
            # The function takes a single Match argument, and returns the replacement string.
            # regex = "\\n"
            regex = re.sub(r"\\x([0-9A-Fa-f]{2})", replace_hex, regex)

            # eliminate mod modifier
            regex = re.sub(r"\(\?.+?\)", "", regex)

            # backreferences
            if re.search(r"\\\d", regex) is not None:
                raise PredictableException("back reference")
            if re.search(r"\\g{", regex) is not None:
                raise PredictableException("back reference")
            if re.search(r"\\k", regex) is not None:
                raise PredictableException("back reference")
            if re.search(r"\(\?P=", regex) is not None:
                raise PredictableException("back reference")

            printable_regex = ""
            for c in regex:
                # will handle both not printable ones and not ascii ones
                if c not in printable:
                    c = random.choice(printable)
                    if c in (".", "+", "*", "?", "^", "$", "(", ")", "[", "]", "{", "}", "|", "/"):
                        c = "\\" + c
                printable_regex += c
            regex = printable_regex

            for not_printable in ("\a", "\b", "\f", "\n", "\r", "\t", "\v"):
                c = random.choice(printable)
                if c in (".", "+", "*", "?", "^", "$", "(", ")", "[", "]", "{", "}", "|", "/"):
                    c = "\\" + c
                regex = regex.replace(not_printable, c)

            # negated character class
            if re.search(r"(?<!\\)\[\^", regex) is not None:
                raise PredictableException("negated character class")

            character_classes = re.finditer(r"(?<!\\)(\[)(.*)(\])", regex)
            for match in character_classes:
                character_class: str = match.group(2)
                alnum_set = "\w" in character_class
                unescaped = character_class.replace("\\", "").replace("-", "")
                sigma_set = not character_class.isalnum()
                if sigma_set:
                    over_approximation = "."
                elif alnum_set or not unescaped.isdigit():
                    over_approximation = "\w"
                else:
                    over_approximation = "\d"
                regex = regex.replace(f"[{character_class}]", over_approximation)

            # look around assertion
            if "(?=" in regex or "(?!" in regex or "(?<=" in regex or "(?<!" in regex:
                raise PredictableException("look around assertion")

            regex = remove_anchor(regex)
            regex = remove_redundant_quantifier(regex)
            regex = preprocess_parenthesis_flag(regex)
            # regex = special_characterize(regex)
            # print("idx:", idx + 1)
            regex = get_captured_regex(regex)

            # redundant escape
            regex = re.sub(r"(?<!\\)/", r"\/", regex)
            regex = re.sub(r"(\\)(&|=)", r"\2", regex)

            # or without left
            regex = re.sub(r"(\()(\|)", r"\1", regex)

            # print("final regex:", regex)
            # print("repre:", repr(regex))

            # if idx == 75:
            #    print("before:", repr(regex))
            # if idx == 75:
            #    print("after:", regex)
            #    exit()
            # print("final:", regex)
            # exit()
            # regex, mapping_table = replace_constant_string(regex)

            if re.search(r"(?<!\x5c)\[[^\[\]]*[()][^\[\]]*\](?!\x5c)", regex) is not None:
                raise PredictableException("overlapped backet")

        except Exception as e:
            # if not isinstance(e, PredictableException) and not isinstance(e, re.error):
            error_idx.append(idx)
            # if data_name[:-3] == "practical_regexes":
            #    print(data_name, idx)
            continue

        try:
            for _ in range(AUGMENTATION_RATIO):
                # generate pos, neg, label
                symbolized_regex, pos, substitutions = make_pos(regex, xeger)

                # print("before removal:", symbolized_regex)
                # symbolized_regex = re.sub(r"(\\)(\x00|\x01|\x02|\x03|\x04|\x05|\x06)", r"\2", symbolized_regex)

                # print("done pos")
                neg = make_neg(symbolized_regex, pos, substitutions)
                # print("done neg")
                label, subregex_list = make_label(symbolized_regex, pos)
                # print("done label")

                for i in range(len(subregex_list)):
                    subregex_list[i] = re.sub("\?P\<t\d*?\>", "", subregex_list[i])

                # \x00 to \x06 are not printable
                # print(repr(regex))
                # print(pos)
                # print(neg)
                # print(label)

                # replace unrecognized symbol
                # print(pos)
                # pos = list(map(lambda y: preprocess_replace(repr(y)[1:-1]), pos))
                # print(pos)
                # print(neg)
                # neg = list(map(lambda y: preprocess_replace(repr(y)[1:-1]), neg))
                # print(neg)
                # exit()

                def tokenize(sequences):
                    if type(sequences) == str:
                        return list(sequences)
                    tokenized = []
                    for sequence in sequences:
                        tokenized.append(list(sequence))
                    return tokenized

                train_pos = pos[: EXAMPLE_NUM // 2]  # list of string
                valid_pos = pos[EXAMPLE_NUM // 2 :]  # list of string
                train_neg = neg[: EXAMPLE_NUM // 2]  # list of string
                valid_neg = neg[EXAMPLE_NUM // 2 :]  # list of string
                labelled_pos = label  # list of string
                symbolized_regex = symbolized_regex  # string

                train_pos = tokenize(train_pos)
                valid_pos = tokenize(valid_pos)
                train_neg = tokenize(train_neg)
                valid_neg = tokenize(valid_neg)
                labelled_pos = tokenize(labelled_pos)
                symbolized_regex = tokenize(symbolized_regex)

                max_len = max(max_len, len(symbolized_regex))
                writer.writerow([train_pos, valid_pos, train_neg, valid_neg, labelled_pos, symbolized_regex])
                """
                writer.writerow([f"{idx} regex"])
                writer.writerow([train_pos])
                writer.writerow([valid_pos])
                writer.writerow([train_neg])
                writer.writerow([valid_neg])
                writer.writerow([labelled_pos])
                writer.writerow([subregex_list])
                """

                """
                total = pos + neg + label
                res = ""
                for ele in total:
                    res = res + str(ele) + ", "
                res = res + str(regex)

                save_file.write(res + "\n")
                """

        except Exception as e:
            # if not isinstance(e, PredictableException) and not isinstance(e, re.error):
            # if e == "NoneType' object has no attribute 'groupdict":
            #    print(e)
            error_idx.append(idx)
            # if data_name[:-3] == "practical_regexes":
            #    print(data_name, idx)
            if str(e) == "can not make EXAMPLE_NUM of examples":
                continue
            if str(e) == "'NoneType' object has no attribute 'groupdict'":
                continue
            # print(e)
            # print(symbolized_regex)
            # print(repr(symbolized_regex))
            # exit()
            continue
        # if data_name[:-3] == "practical_regexes":
        #    print(data_name, idx)
        # if idx % 1000 == 0:
        #    print(idx)
    save_file.close()
    print(f"data_name: {data_name}, max_len: {max_len}", data_name)
    print("error count :", len(error_idx))
    print("total len:", len(regex_list))
    print("max_len:", max_len)


if __name__ == "__main__":
    main()
