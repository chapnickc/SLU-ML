import numpy as np
import pandas as pd
from collections import Counter
import json


SPECIAL_CHARS = {
    'ç': 'c', 'Ç': 'C',
    'ğ':'g',  'Ğ':'G',
    'ö': 'o', 'Ö': 'O',
    'ş':'s',  'Ş':'S',
    'ü': 'u', 'Ü': 'U',
    'ı': 'i', 'İ': 'I',
    'ə': 'e', 'Ə': 'E'
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


# remove special characters
unspecial = []
for token in lines:
    key = token
    for char, repl in SPECIAL_CHARS.items():
        key = key.replace(char, repl)
    unspecial.append(key)


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


def best_form(token):
    forms = token_map.get(token)
    if forms:
        repl, _ = max(forms.items(), key=lambda x: x[1])
        return repl
    else:
        return token


# with open('token_map.json', 'w') as f: json.dump(token_map, f)

test_df = pd.read_csv('./input.csv')
df = test_df.copy()

df['token'] = df['token'].apply(lambda tok: best_form(tok))

df.to_csv('output.csv', index=False)
