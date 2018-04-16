import os.path
import collections
from operator import itemgetter

from nltk import ngrams

def ngrams(word, ngram_size=3):
    "Given a word, return the set of unique ngrams in that word."
    all_ngrams = set()
    for i in range(0, len(word) - ngram_size + 1):
        all_ngrams.add(word[i:i + ngram_size])
    return all_ngrams



def suggested_words(ngram_words, target_word, results=5, len_variance=0):
    "Given a word, return a list of possible corrections."
    word_ranking = collections.defaultdict(int)
    possible_words = set()
    for ngram in ngrams(target_word):
        words = ngram_words[ngram]
        for word in words:
            # only use words that are within +-LEN_VARIANCE characters in 
            # length of the target word
            if len(word) >= len(target_word) - len_variance and \
               len(word) <= len(target_word) + len_variance:
                word_ranking[word] += 1
    # sort by descending frequency 
    ranked_word_pairs = sorted(word_ranking.items(), key=itemgetter(1), reverse=True)
    return [word_pair[0] for word_pair in ranked_word_pairs[0:results]]



def best_form(autocorrect_obj, token):
    try:
        assert type(token) == str
    except AssertionError:
        return token

    forms = autocorrect_obj.suggested_words(token)
    if len(forms) > 0:
        return forms[0]
    else:
        return token




if __name__ == '__main__':
    import pandas as pd

    autocorrect = Autocorrect(len_variance=0)


    test_df = pd.read_csv('./input.csv')
    df = test_df.copy()

    df['token'] = df['token'].apply(lambda tok: best_form(autocorrect, tok))

    df.to_csv('output_autocorrect.csv', index=False)

    #-------------------------------------------------------
    ngram_size = 3
    len_variance = 0


    teststr = 'sozun' #'sözün'
    teststr = 'formalarindan' #'formalarından'
    teststr = 'istifade' #'istifadə'


    wordfile = './azj-train.txt'

    # self.words = set(open(wordfile).read().splitlines())
    with open(wordfile) as f:
        words = set([l.rstrip() for l in f.readlines()])

    # create dictionary of ngrams and the words that contain them
    ngram_words = collections.defaultdict(set)
    for word in words:
        for ngram in ngrams(word):
            ngram_words[ngram].add(word)
    print("Generated %d ngrams from %d words" % (len(ngram_words), len(words)))






