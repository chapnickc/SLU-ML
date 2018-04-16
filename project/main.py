import os
import glob
import warnings

import numpy as np
import pandas as pd

import librosa
import librosa.display
import audioread

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm

from matplotlib import pyplot as plt
from matplotlib.pyplot import specgram

from utils import *



data_dir='./Cat-Dog/cats_dogs/'

cat_files = glob.glob(data_dir + 'cat*')
dog_files = glob.glob(data_dir + 'dog*')

cat_files.sort(key = lambda f: get_file_num(f))
dog_files.sort(key = lambda f: get_file_num(f))

cats = load_data(cat_files, sr=16000)
dogs = load_data(dog_files, sr=16000)

cats_mel_frequencies = compute_mel_frequencies(cats, sr=16000)
dogs_mel_frequencies = compute_mel_frequencies(dogs, sr=16000)

# 20 mfccs bins (rows) and N columns depending on len of signal
cats_mfccs = compute_mfccs(cats_mel_frequencies)
dogs_mfccs = compute_mfccs(dogs_mel_frequencies)

# cats_deltas = compute_deltas(cats_mfccs)
# dogs_deltas = compute_deltas(dogs_mfccs)
new_cats_mfccs = remove_frames(cats, cats_mfccs)
new_dogs_mfccs = remove_frames(dogs, dogs_mfccs)

#-----------------------------------------------
from notmine import split_data
df_train, df_test = split_data(new_cats_mfccs, new_dogs_mfccs, len(cats), len(dogs), test_size=0.3)


X_train = df_train.iloc[:, :-1]
y_train = df_train.iloc[:, -1]

X_test = df_test.iloc[:, :-1]
y_test = df_test.iloc[:, -1]


#-----------------------------------------------


# build the training dataframe
df_cats = build_df_mfccs(new_cats_mfccs, label=0)
df_dogs = build_df_mfccs(new_dogs_mfccs, label=1)
df = pd.concat([df_cats, df_dogs.set_index(df_dogs.index + df_cats.index[-1])])

df = df.sample(frac=1, random_state=42)


# labels=df['Label']
# labels = one_hot_encode(labels)

features = df.iloc[:,:-1]
labels = df.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

df_train = pd.concat([X_train, y_train], axis=1)
df_test = pd.concat([X_test, y_test], axis=1)
#------------------------------------------------------------------

logreg = LogisticRegression()
logreg.fit(X_train, y_train)


## Test accuracy
y_pred_test = logreg.predict(X_test)

# aggregate according to the original audio file
# label 1 if average label is >= 0.5
pred_test_boolean = (pd.Series(y_pred_test, index=y_test.index)
                        .groupby(y_test.index)
                        .mean()) >= 0.5
y_pred_test_last= pred_test_boolean*1

# labels for each audio file
y_test_last = df_test['Label'].groupby(y_test.index).mean()

test_accuracy = np.sum(np.array([y_test_last == y_pred_test_last]))/len(y_test_last)

## Training accuracy
y_pred_train = logreg.predict(X_train)
pred_train_boolean = (pd.Series(y_pred_train, index=y_train.index)
                        .groupby(y_train.index)
                        .mean()) >= 0.5
y_pred_train_last= pred_train_boolean*1

# labels for each audio file
y_train_last = df_train['Label'].groupby(y_train.index).mean()

train_accuracy = np.sum(np.array([y_train_last == y_pred_train_last]))/len(y_train_last)





print('\n\nTest Set:\n')
print('Test Confusion Matrix:')
print(confusion_matrix(y_test_last, y_pred_test_last))
np.sum(confusion_matrix(y_test_last, y_pred_test_last))

print('Test Classification report:')
print(classification_report(y_test_last, y_pred_test_last))
print('Test Accuracy: ' + str(test_accuracy))

print('\n\nTraining Set:\n')
print('Training Confusion Matrix:')
print(confusion_matrix(y_train_last, y_pred_train_last))
np.sum(confusion_matrix(y_train_last, y_pred_train_last))

print('Train Classification report:')
print(classification_report(y_train_last, y_pred_train_last))
print('Train Accuracy: ' + str(train_accuracy))

