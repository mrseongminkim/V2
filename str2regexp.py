import string
import lark
from FAdo.reex import RegExp, BuildRegexp

# 33 + 1
punctuation = string.printable[-38:-5]
idx = 128
p2s = {}
s2p = {}
for punct in punctuation:
    symbol = chr(idx)
    p2s[punct] = symbol
    s2p[symbol] = punct
    idx += 1
# handle <unk>
symbol = chr(idx)
p2s["<unk>"] = symbol
s2p[symbol] = "<unk>"

regGrammar = lark.Lark.open("ascii_regexp_grammar.lark", rel_to=__file__, start="rege", parser="lalr")


def str2regexp(s):
    tree = regGrammar.parse(s)
    reg = RegExp()
    reg = BuildRegexp(context={"sigma": None}).transform(tree)
    return reg
