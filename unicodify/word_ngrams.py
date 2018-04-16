import numpy as np
import pandas as pd
from collections import Counter
import json

# import string
from nltk import ngrams


SPECIAL_CHARS = {
#     'À':'A', 'Â':'A', 'Ä':'A',
    # 'à':'a', 'á':'a', 'â':'a',
    # 'ã':'a', 'ä':'a', 'å':'a',
    # 'ā':'a', 'ą':'a',
    # 'Ð':'D',
    # 'è':'e', 'é':'e', 'ê':'e',
    # 'Ģ':'G',
    # 'ì':'i', 'í':'i', 'î':'i', 'ï':'i', 'ī':'i',
    # 'ò':'o', 'ó':'o',
    # 'Û':'U', 'û': 'u', 'ü': 'u',
    # 'к': 'k',
    # 'Ň': 'n', 'ñ': 'n', 'ň': 'n', 'ŋ': 'n', 'η': 'n',
    # 'Ý': 'Y', 'ý': 'y',
    # 'ÿ': 'y',
    # 'Ž': 'Z', 'ž': 'z',
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


fname = './azj-train.txt'
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


trigrams = list(ngrams(lines, 3))
trigram_counts = Counter(trigrams)

bigrams = list(ngrams(lines, 2))
bigram_counts = Counter(bigrams)





def unigram_estimate(token):
    forms = token_map.get(token)
    if forms:
        repl, _ = max(forms.items(), key=lambda x: x[1])
        return repl
    else:
        return token


def trigram_estimate(token, previous_words):
    forms = token_map.get(token)
    if forms:
        for form in forms:
            key = tuple([ *list(previous_words), form ])
            key in trigram_counts.keys()

            trigrams_counts[key]



        # repl, _ = max(forms.items(), key=lambda x: x[1])
        return repl
    else:
        return token



# with open('token_map.json', 'w') as f: json.dump(token_map, f)

test_df = pd.read_csv('./input.csv')
df = test_df.copy()


i = 0
df.at[i, 'token'] = unigram_estimate(df['token'].at[i])

i = 1
df.at[i, 'token'] = unigram_estimate(df['token'].at[i])



for i in range(2, len(df.rows)):
    token = df.loc[i, 'token']
    previous_words = df.loc[i-2:i-1, 'token']






df['token'] = df['token'].apply(lambda tok: best_form(tok))

# df.to_csv('output.csv', index=False)
df.to_csv('output_test.csv', index=False)


sorted_trigrams = sorted(trigram_counts.items(), key=lambda x: x[1], reverse=True)




#-----------------------------------



# char_seqs = [c for tok in lines for c in tok]

# trigrams = list(ngrams(char_seqs, 3))
# trigram_counts = Counter(trigrams)

# bigrams = list(ngrams(char_seqs, 2))
# bigram_counts = Counter(bigrams)


# unique_chars = set(char_seqs)
# SPECIAL_CHARS = set([c for c in unique_chars if not c in string.printable])
# SPECIAL_CHARS = sorted(list(SPECIAL_CHARS))
# string.ascii_letters


# test_df = pd.read_csv('./input.csv')
# df = test_df.copy()
# char_seqs = [c for tok in df['token'] if type(tok) == str for c in tok]
# SPECIAL_CHARS = [c for c in unique_chars if not c in string.printable]



