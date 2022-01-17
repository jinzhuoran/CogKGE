# from fast_fuzzy_search import FastFuzzySearch
#
# ffs = FastFuzzySearch({'language':'english'})
# ffs.add_term('hello world', 0)
# ffs.add_term('hello friend', 1)
# ffs.add_term('helu world', 2)
# ffs.add_term('helwerwwerwerwererwerud', 3)
# print(123)
# ffs.add_term('2015MaltaBadmintonChampionships', 4)
# print(456)
# results = ffs.search('helu world')
#
# print(results)
#
# # [(id, text, score)]

# import re
# table = ''.maketrans('1234', 'abcd')
# s2 = '1334r5254523343'
# print(s2.translate(table))
import copy
xx={12:34}
y=copy.deepcopy(xx)
y[12]=99
print(xx)
print(y)