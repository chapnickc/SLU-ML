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




# 'umumi': {'umumi': 8, 'ümumi': 1170, 'ümümi': 2},
# key='umumi'
# forms = token_map[key]
# word = list(forms.keys())[0]
# count = forms[word]

from tqdm import tqdm

token_to_ngrams = {}

for key, forms in tqdm(token_map.items()):
    token_to_ngrams[key] = {}
    token_to_ngrams[key]['first_letter'] = []
    token_to_ngrams[key]['bigrams'] = []
    token_to_ngrams[key]['trigrams'] = []

    for word, count in forms.items():
        token_to_ngrams[key]['first_letter'].extend(list(word[0])*count)

        bigrams = list(ngrams(list(word), n=2))
        token_to_ngrams[key]['bigrams'].extend(bigrams*count)

        trigrams = list(ngrams(list(word), n=3))
        token_to_ngrams[key]['trigrams'].extend(trigrams*count)

    token_to_ngrams[key]['first_letter'] = Counter(token_to_ngrams[key]['first_letter'])
    token_to_ngrams[key]['bigrams'] = Counter(token_to_ngrams[key]['bigrams'])
    token_to_ngrams[key]['trigrams'] = Counter(token_to_ngrams[key]['trigrams'])






def best_form(key):
    print(key)
    try:
        assert type(key) == str
    except AssertionError:
        return key

    if not token_map.get(key):
        return key

    if len(token_map.get(key)) == 1:
        return list(token_map.get(key).keys())[0]

    repl = ''
    for i in range(len(key)):
        if i == 0:
            next_char = token_to_ngrams[key]['first_letter'].most_common(1)[0][0]
            repl += next_char
        elif i == 1:
            previous_char = repl[i-1]
            acceptable_chars = [key[i], *list(k for k,v in SPECIAL_CHARS.items() if v == key[i])]
            cantidates = {  gram : count
                            for gram,count in token_to_ngrams[key]['bigrams'].items()
                            if gram[0] == previous_char and gram[1] in acceptable_chars
                         }
            next_char = max(cantidates.items(), key=lambda x: x[1])[0][-1]
            repl += next_char
        elif i >= 2:
            previous_chars = tuple( repl[i-2:i] )
            acceptable_chars = [key[i], *list(k for k,v in SPECIAL_CHARS.items() if v == key[i])]
            cantidates = {  gram : count
                            for gram,count in token_to_ngrams[key]['trigrams'].items()
                            if gram[:2] == previous_chars and gram[-1] in acceptable_chars
                         }
            next_char = max(cantidates.items(), key=lambda x: x[1])[0][-1]
            repl += next_char
    return repl



best_form(str(df.iloc[4, 1]))


key = 'yasayan'
best_form('yasayan')
token_map[key]



    print(word, count)

    key = remove_special_chars(variant)
    if token_map.get(key):
        bigrams = list(ngrams(list(variant), n=2))
        trigrams = list(ngrams(list(variant), n=3))
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



# with open('token_map.json', 'w') as f: json.dump(token_map, f)

i = 0
df.at[i, 'token'] = unigram_estimate(df['token'].at[i])

i = 1
df.at[i, 'token'] = unigram_estimate(df['token'].at[i])



for i in range(2, len(df.rows)):
    token = df.loc[i, 'token']
    previous_words = df.loc[i-2:i-1, 'token']




test_df = pd.read_csv('../input.csv')
df = test_df.copy()



df['token'] = df['token'].apply(lambda tok: best_form(tok))

# df.to_csv('output.csv', index=False)
df.to_csv('../output/output_trigram_letters.csv', index=False)


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



