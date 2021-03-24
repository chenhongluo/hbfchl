import re

re_digits = re.compile(r'(\d+)')


def emb_numbers(s):
    pieces = re_digits.split(s)
    pieces[1::2] = map(int, pieces[1::2])
    return pieces


def sort_strings_with_emb_numbers(alist):
    aux = [(emb_numbers(s), s) for s in alist]
    aux.sort()
    return [s for __, s in aux]


def sort_strings_with_emb_numbers2(alist):
    return sorted(alist, key=emb_numbers)

