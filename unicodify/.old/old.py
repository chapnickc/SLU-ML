import pandas as pd
from collections import Counter

SPECIAL_CHARS = {'ç': 'c', 'ə':'e', 'ğ':'g', 'ı':'i', 'ö':'o', 'ş': 's', 'ü': 'u'}

df = pd.read_csv('./azj-train.txt', sep=' ')

tokens = df.as_matrix().tolist()

tokens = [x[0] for x in tokens]
token_count = dict(Counter(tokens))

for token, count in token_count.items():

    if type(token) == float:
        print(token, count)

    if type(token) == str:
        key = token
        for char, repl in SPECIAL_CHARS.items():
            if char in key:
                key = key.replace(char, repl)



all(type(x) == str for x in token_count)
k = [x for x in token_count if type(x) == float][0]

token_count.get(k)


token = 'soərğuya'








