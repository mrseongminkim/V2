import sre_parse
from random import randint
from typing import List

# We don't actually have to escape {, }, and ] but... who knows
SPECIAL_SYMBOLS = "$()*+./?[\\]^|{}"
DEAD_OPCDE = [sre_parse.GROUPREF, sre_parse.ASSERT, sre_parse.ASSERT_NOT, sre_parse.GROUPREF_EXISTS, sre_parse.NOT_LITERAL, sre_parse.NEGATE]
REPEAT = [sre_parse.MAX_REPEAT, sre_parse.MIN_REPEAT]
NEGATION = [sre_parse.NEGATE, sre_parse.NOT_LITERAL, sre_parse.CATEGORY_NOT_DIGIT, sre_parse.CATEGORY_NOT_SPACE, sre_parse.CATEGORY_NOT_WORD]


def is_punct(min: int, max: int) -> bool:
    if min < ord("0") or max > ord("z"):
        return True
    if (ord(":") <= min and min <= ord("@")) or (ord("[") <= min and min <= ord("`")):
        return True
    if (ord(":") <= max and max <= ord("@")) or (ord("[") <= max and max <= ord("`")):
        return True
    return False


def is_alpha(min: int, max: int) -> bool:
    if (ord("A") <= min and min <= ord("Z")) and (ord("A") <= max and max <= ord("Z")):
        return True
    if (ord("a") <= min and min <= ord("z")) and (ord("a") <= max and max <= ord("z")):
        return True
    return False


def get_symbol(argument: int) -> str:
    # Handle non-printable characters and non-ascii characters
    if argument < ord(" ") or argument > ord("~"):
        argument = randint(ord(" "), ord("~"))
    symbol = chr(argument)
    # Since argument is literal, we need to escape it
    if symbol in SPECIAL_SYMBOLS:
        symbol = "\\" + symbol
    return symbol


def preprocess_regex(ast: List[tuple], is_root: bool = True, generatable=[False, 1]) -> str:
    regex = "("
    prev_opcode = None
    for opcode, argument in ast:
        assert opcode not in DEAD_OPCDE
        if opcode == sre_parse.AT:
            continue

        if prev_opcode and ((prev_opcode != opcode) or (prev_opcode in REPEAT and opcode in REPEAT) or (prev_opcode == sre_parse.SUBPATTERN and opcode == sre_parse.SUBPATTERN)):
            regex += ")("
        prev_opcode = opcode

        if opcode == sre_parse.LITERAL:
            symbol = get_symbol(argument)
            regex += symbol
        elif opcode in REPEAT:
            min, max, argument = argument
            sub_regex = preprocess_regex(argument, is_root=False, generatable=generatable)
            if min == 0 and max == 1:
                operator = "?"
            else:
                operator = "*"
                generatable[0] = True
            if sub_regex[-1] == operator:
                operator = ""
            if is_root:
                regex += sub_regex + operator
            else:
                return sub_regex + operator
        elif opcode == sre_parse.IN:
            generatable[0] = True
            # Negation always comes first
            assert argument[0][0] not in NEGATION
            # Categories
            if len(argument) == 1 and argument[0][0] == sre_parse.CATEGORY:
                argument = argument[0][1]
                assert argument not in NEGATION
                if argument == sre_parse.CATEGORY_DIGIT:
                    character_class = "\d"
                elif argument == sre_parse.CATEGORY_WORD:
                    character_class = "\w"
                elif argument == sre_parse.CATEGORY_SPACE:
                    character_class = "."
            else:
                character_class = "\d"
                for sub_opcode, sub_argument in argument:
                    if sub_opcode == sre_parse.RANGE:
                        min, max = sub_argument
                        if is_punct(min, max):
                            character_class = "."
                            break
                        elif is_alpha(min, max):
                            character_class = "\w"
                    if sub_opcode == sre_parse.CATEGORY:
                        assert sub_argument not in NEGATION
                        if sub_argument == sre_parse.CATEGORY_SPACE:
                            character_class = "."
                            break
                        elif sub_argument == sre_parse.CATEGORY_WORD:
                            character_class = "\w"
                    if sub_opcode == sre_parse.LITERAL:
                        if is_punct(sub_argument, sub_argument):
                            character_class = "."
                            break
                        elif is_alpha(sub_argument, sub_argument):
                            character_class = "\w"
            if is_root:
                regex += character_class
            else:
                return character_class
        elif opcode == sre_parse.ANY:
            generatable[0] = True
            if is_root:
                regex += "."
            else:
                return "."
        elif opcode == sre_parse.SUBPATTERN:
            group, add_flags, del_flags, argument = argument
            sub_regex = preprocess_regex(argument, is_root=False, generatable=generatable)
            if is_root:
                regex += sub_regex
            else:
                return sub_regex
        elif opcode == sre_parse.BRANCH:
            sub_regex = ""
            sub_regex_set = set()
            _, argument = argument
            for sub_pattern in argument:
                temp = preprocess_regex(sub_pattern, is_root=False, generatable=generatable)
                if temp == "()":
                    continue
                elif temp == ".*":
                    sub_regex = temp
                    break
                else:
                    sub_regex_set.add(temp)
            if sub_regex == "":
                generatable[1] *= len(sub_regex_set)
                if generatable[1] >= 10:
                    generatable[0] = True
                sub_regex = "|".join(sub_regex_set)
            sub_regex = "(" + sub_regex + ")"
            if is_root:
                regex += sub_regex
            else:
                return sub_regex
        else:
            print(opcode)
            exit()
    return regex + ")", generatable[0]
