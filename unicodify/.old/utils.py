import numpy as np
from collections import Counter

SPECIAL_CHARS = {
    'ç': 'c',
    'ğ':'g',
    'ö': 'o',
    'ş':'s',
    'ü': 'u',
    'ı': 'i',
    'ə': 'e'
}

def remove_special_chars(token):
    key = token
    for char, repl in SPECIAL_CHARS.items():
        key = key.replace(char, repl)
    return key


fname = './azj-train.txt'
#def read_sentences():

with open(fname) as f:
    lines = [l.rstrip() for l in f.readlines()]


indices = [
        i for i, l in enumerate(lines)
        if l == "." and not(lines[i-1] == '.' and lines[i+1] == '.')
    ]


# create a list of all the sentences
lastix = 0
sentences=[]
for i in indices:
    s = lines[lastix: i + 1]
    sentences.append(s)
    lastix = i + 1

lengths = [len(s) for s in sentences]

len_two = [i for i, s in enumerate(sentences) if len(s) == 2]

# remove special characters
unspecial = []
for token in lines:
    key = token
    for char, repl in SPECIAL_CHARS.items():
        key = key.replace(char, repl)
    unspecial.append(key)



# build a dict where each key is the unspecial version
# then the values are all the variations with their counts

unspecial
counts = Counter(lines)

token_map = {}

for variant, value in counts.items():
    key = remove_special_chars(variant)
    if token_map.get(key):
        token_map[key][variant] = value
    else:
        token_map[key] = {}
        token_map[key][variant] = value




