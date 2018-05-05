import numpy as np
import pandas as pd
from collections import Counter
import json

import nltk
from nltk import ngrams
from tqdm import tqdm

nltk.download('punkt')



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


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]



fname = '../azj-train.txt'
with open(fname) as f:
    lines = [l.rstrip() for l in f.readlines()]


fname = '../wiki.aze.txt'
with open(fname) as f:
    wiki = f.read()
wiki = wiki.strip('\n')
wiki = wiki.strip('\n\n')


fname = '../web.aze.txt'
with open(fname) as f:
    web = f.read()
web = web.strip('\n')
web = web.strip('\n\n')


all_tokens = []
for chunk in tqdm(chunks(wiki, round(30010))):
    tokens = nltk.word_tokenize(chunk)
    all_tokens.extend(tokens)

all_tokens.extend(lines)

new_tokens = []
for chunk in tqdm(chunks(web, round(1e6))):
    tokens = nltk.word_tokenize(chunk)
    new_tokens.extend(tokens)

all_tokens.extend(new_tokens)

N = len(lines)
V = len(set(lines))


# build a dict where each key is the unspecial version
# then the values are all the variations with their counts
counts = Counter(all_tokens)

token_map = {}
for variant, value in counts.items():
    key = remove_special_chars(variant)
    if token_map.get(key):
        token_map[key][variant] = value
    else:
        token_map[key] = {}
        token_map[key][variant] = value


def unigram_estimate(token):
    forms = token_map.get(token)
    if forms:
        repl, _ = max(forms.items(), key=lambda x: x[1])
        return repl
    else:
        return token


with open('token_map.json', 'w') as f:
    json.dump(token_map, f)

test_df = pd.read_csv('../input.csv')
df = test_df.copy()

df['token'] = df['token'].apply(lambda tok: unigram_estimate(tok))

df.to_csv('output.csv', index=False)
# df.to_csv('../output/output_web_wiki_and_train.csv', index=False)
