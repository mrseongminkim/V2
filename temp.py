'''
# https://stackoverflow.com/questions/40556491/how-to-find-the-longest-common-substring-of-multiple-strings
from functools import partial, reduce
from itertools import chain
from typing import Iterator


def ngram(seq: str, n: int) -> Iterator[str]:
    return (seq[i: i+n] for i in range(0, len(seq) - n + 1))


def allngram(seq: str) -> set:
    lengths = range(len(seq))
    ngrams = map(partial(ngram, seq), lengths)
    return set(chain.from_iterable(ngrams))


sequences = ['/7=&AnSSip=', '/0=&AnSSip=', '/9=&AnSSip=', '/1=&AnSSip=', '/5=0&AnSSip=', '/1=1&AnSSip=', '/8=&AnSSip=', '/4=543&AnSSip=', '/3=808&AnSSip=', '/5=7&AnSSip=', '/6=0&AnSSip=', '/3=0&AnSSip=', '/5=&AnSSip=', '/8=4&AnSSip=', '/6=02&AnSSip=', '/4=4&AnSSip=', '/2=096&AnSSip=', '/5=597&AnSSip=', '/4=12&AnSSip=', '/6=181&AnSSip=']
seqs_ngrams = map(allngram, sequences)
intersection = reduce(set.intersection, seqs_ngrams)
longest = max(intersection, key=len)
'''

def get_n_grams(string: str, n: int) -> set:
    n_grams = set()
    for i in range(0, len(string) - n + 1):
        n_grams.add(string[i : i + n])
    return n_grams

def get_all_grams(string: str) -> set:
    all_grams = set()
    for length in range(len(string)):
        n_grams = get_n_grams(string, length)
        all_grams.update(n_grams)
    return all_grams

sequences = ['/7=&AnSSip=', '/0=&AnSSip=', '/9=&AnSSip=', '/1=&AnSSip=', '/5=0&AnSSip=', '/1=1&AnSSip=', '/8=&AnSSip=', '/4=543&AnSSip=', '/3=808&AnSSip=', '/5=7&AnSSip=', '/6=0&AnSSip=', '/3=0&AnSSip=', '/5=&AnSSip=', '/8=4&AnSSip=', '/6=02&AnSSip=', '/4=4&AnSSip=', '/2=096&AnSSip=', '/5=597&AnSSip=', '/4=12&AnSSip=', '/6=181&AnSSip=']

sequences_n_grams = set()
for sequence in sequences:
    all_grams = get_all_grams(sequence)
    sequences_n_grams.add(frozenset(all_grams))
intersection = frozenset.intersection(*sequences_n_grams)
longest_common_substring = max(intersection, key=len)
print(longest_common_substring)
