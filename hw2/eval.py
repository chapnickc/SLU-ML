
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer


def euclid_distance(x1, x2):
    return np.sqrt(sum((x2 - x1)**2))


def knn_predict(D, labels, k, x):
    S = []
    for n, xn  in enumerate(D):
        d = (euclid_distance(xn, x), n)
        S.append(d)
    S.sort(key=lambda x: x[0])
    y = 0
    for k in range(k):
        _, n = S[k]
        y += labels[n]
    return np.sign(y)


if __name__ == "__main__":
    df = load_breast_cancer()

    df.target[df.target == 0] = -1

    ix = round(0.80*df.data.shape[0])

    df_train = df.data[:ix, :]
    df_train_labels = df.target[:ix]

    df_test = df.data[ix:, :]
    df_test_labels = df.target[ix:]

    x = df_test[0,:]
    df_test_labels[0]

    success = 0
    for (i, x_test) in enumerate(df_test):
        y_p = knn_predict(df_train, df_train_labels, k=5, x=x_test)
        if y_p == df_test_labels[i]:
            success += 1
    success/len(df_test) * 100

