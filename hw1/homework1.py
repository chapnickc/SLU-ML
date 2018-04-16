from collections import Counter
import random
import pandas as pd
import numpy as np


class Leaf(object):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return str(self.value)


class Node(object):
    def __init__(self, feature, left, right, score=None):
        self.feature = feature
        self.left = left
        self.right = right
        self.score = score

    def __repr__(self):
        return f'Node({str(self.feature)}, {self.left}, {self.right})'


def entropy(D):
    """Compute the entropy of a distribution"""
    counts = dict(Counter(D))
    total = sum(counts.values())
    H = sum( -(n/total) * np.log2(n/total) for n in counts.values() )
    return H


def feature_entropy(feature, ratings):
    Y = N = np.array([])
    for answer, rating in zip(feature, ratings):
        if answer == 1:
            Y = np.append(Y, rating)
        elif answer == 0:
            N = np.append(N, rating)
    total = len(N) + len(Y)
    H = sum( len(D) / total * entropy(D) for D in [N, Y] )
    return H


def decision_tree_train(data, remaining_features):
    try:
        # most frequent answer in data
        guess, _ = Counter(data).most_common(1)[0]
    except IndexError:
        guess = random.getrandbits(1)

    if all(data == guess):
        return Leaf(guess)
    elif remaining_features.empty:
        return Leaf(guess)

    score = {}
    for f in remaining_features:
        NO = data[remaining_features[f] == 0]
        YES = data[remaining_features[f] == 1]
        H = feature_entropy(remaining_features[f], data)
        score[f] = H

    # best score is the one with lowest entropy
    best_feature = min(score, key=score.get)
    best_score =score[best_feature]

    NO = data[remaining_features[best_feature] == 0]
    YES = data[remaining_features[best_feature] == 1]

    NO_features = (remaining_features[remaining_features[best_feature] == 0]
        .drop(best_feature, axis=1))
    YES_features = (remaining_features[remaining_features[best_feature] == 1]
        .drop(best_feature, axis=1))

    left = decision_tree_train(NO, NO_features)
    right = decision_tree_train(YES, YES_features)
    return Node(best_feature, left, right, best_score)



def decision_tree_test(tree, test_point):
    pass


if __name__ == '__main__':

    df = pd.read_csv('chapter1_dataset.csv', index_col=0)

    data = df.iloc[:,-1]
    data[data >= 0] = 1
    data[data < 0] = 0

    remaining_features = df.iloc[:,:-1]

    tree = decision_tree_train(data, remaining_features)
    print(tree)
