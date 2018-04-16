char_set = 'abcdefghijklmnopqrstuvwxyz'

def edits1(word):
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    inserts = [
            ('ins', a[-1] + c, a[1:] + c + b)
            for a, b in map(lambda e: ('@' + e[0], e[1]), splits) 
            for c in char_set
        ]
    deletes = [
            ('del', a[-1] + b[0], a[1:] + b[1:])
            for a, b in map(lambda e: ('@' + e[0], e[1]), splits) if b
        ]
    substitutions = [
            ('sub', b[0] + c, a + c + b[1:]) 
            for a, b in splits for c in char_set if b
        ]
    transposes = [
            ('trans', b[0] + b[1], a + b[1] + b[0] + b[2:])
            for a, b in splits if len(b) > 1
        ]
    return set(deletes + transposes + substitutions + inserts)


def edits(word, dist):
    if dist == 1:
        return edits1(word)
    return set(e for operation, confusion_params, generated_word in edits(word, dist - 1) for e in edits1(generated_word))


#----------------------------------------------------------------------------------
import re
import collections


fname = './azj-train.txt'
with open(fname) as f:
    lines = [l.rstrip() for l in f.readlines()]

char_seqs = [c for tok in lines for c in tok]


# format the corpus as a list of lowercase words
def features(text):
	return re.findall('[a-z-\']+', text.lower())

# produce a word -> count mapping structure
def train(features):
	bag_of_words = collections.defaultdict(lambda: 1)
	for f in features:
		bag_of_words[f] += 1
	return bag_of_words

bag_of_words = train(char_seqs)

char_set = ''.join(bag_of_words.keys())
char_set = 'abcdefghijklmnopqrstuvwxyz-\''



# generate all the words with edit distance 1
def edits1(word):
	splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
	deletes = [a + b[1:] for a, b in splits if b]
	transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
	replaces = [a + c + b[1:] for a, b in splits for c in char_set if b]
	inserts = [a + c + b for a, b in splits for c in char_set]
	return set(deletes + transposes + replaces + inserts)

# generate all the words with edit distance 2
def edits2(word):
	return set(e2 for e1 in edits1(word) for e2 in edits1(e1))

# generate all the English words (in our corpus) with edit distance 2
def known_edits2(word):
	return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in bag_of_words )

# filter a set of words so that they're all English words (in our corpus)
def known(words):
	return set(w for w in words if w in bag_of_words)

# correct the spelling of a word
def correct(word):
	candidates = known([word]) or known(edits1(word)) or known_edits2(word) or [word]
	return max(candidates, key=bag_of_words.get)






teststr = 'sozun' #'sözün'
teststr = 'formalarindan' #'formalarından'
teststr = 'istifade' #'istifadə'

correct(teststr)
