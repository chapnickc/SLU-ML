import numpy as np
import pandas as pd
from collections import Counter
import json

# import string
from nltk import ngrams


SPECIAL_CHARS = {
    'ç': 'c', 'Ç': 'C',
    'ğ':'g',  'Ğ':'G',
    'ö': 'o', 'Ö': 'O',
    'ş':'s',  'Ş':'S',
    'ü': 'u', 'Ü': 'U',
    'ı': 'i', 'İ': 'I',
    'ə': 'e', 'Ə': 'E',
}

def remove_special_chars(token):
    key = token
    for char, repl in SPECIAL_CHARS.items():
        key = key.replace(char, repl)
    return key

def add_special_chars(token):
    if type(token) != str:
        return token
    key = token
    for repl, char in SPECIAL_CHARS.items():
        key = key.replace(char, repl)
    return key


fname = '../azj-train.txt'
with open(fname) as f:
    lines = [l.rstrip() for l in f.readlines()]

N = len(lines)
V = len(set(lines))



# build a dict where each key is the unspecial version
# then the values are all the variations with their counts
counts = Counter(lines)
token_map = {}
for variant, value in counts.items():
    key = remove_special_chars(variant)
    if token_map.get(key):
        token_map[key][variant] = value
    else:
        token_map[key] = {}
        token_map[key][variant] = value



unigrams=Counter(ngrams(lines, n=1))
bigrams=Counter(ngrams(lines, n=2))
trigrams=Counter(ngrams(lines, n=3))



def unigram_estimate(token):
    forms = token_map.get(token)
    if forms:
        repl, _ = max(forms.items(), key=lambda x: x[1])
        return repl
    else:
        return token



def trigram_estimate(previous_words, token):
    cantidates = token_map.get(token)
    if not cantidates:
        return token


    d = {}
    for k, v in cantidates.items():
        gram = tuple((*previous_words, k))
        d[gram] = trigrams[gram]

    gram, count = max(d.items(), key=lambda x:x[1])
    if count > 0:
        return gram[-1]


    d = {}
    for k, v in cantidates.items():
        gram = tuple((previous_words[-1], k))
        d[gram] = bigrams[gram]

    gram, count = max(d.items(), key=lambda x:x[1])
    if count > 0:
        return gram[-1]

    return unigram_estimate(k)




for i in range(2, len(df.rows)):
    token = df.loc[i, 'token']
    previous_words = tuple(df.loc[i-2:i-1, 'token'])




    d = {}
    for gram,count in trigrams.items():
        if gram[:2] == previous_words:
            d[gram] = count


    cantidates = {  gram : count
                for gram,count in trigrams.items()
                if gram[:2] == previous_words
            }


test_df = pd.read_csv('../input.csv')
df = test_df.copy()

i = 0
df.at[i, 'token'] = unigram_estimate(df['token'].at[i])

i = 1
df.at[i, 'token'] = unigram_estimate(df['token'].at[i])




df['token'] = df['token'].apply(lambda tok: unigram_estimate(tok))

# df.to_csv('output.csv', index=False)
df.to_csv('../output/output_test.csv', index=False)

#--------------------------------------------
# cantidates = [
            # remove_special_chars(word[0])
            # for word, count in unigrams.items()
            # if type(word[0]) == str
#         ]

# len(cantidates)
# len(set(cantidates))


