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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import svm

from matplotlib import pyplot as plt
from matplotlib.pyplot import specgram

from utils import *
from notmine import split_data


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

new_cats_mfccs = remove_frames(cats, cats_mfccs)
new_dogs_mfccs = remove_frames(dogs, dogs_mfccs)

#-----------------------------------------------

df_train, df_test = split_data(new_cats_mfccs, new_dogs_mfccs, len(cats), len(dogs), test_size=0.3)


X_train = df_train.iloc[:, :-1]
y_train = df_train.iloc[:, -1]

X_test = df_test.iloc[:, :-1]
y_test = df_test.iloc[:, -1]

#----------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
#----------------------------------------------------

logreg = LogisticRegression()
logreg.fit(X_train, y_train)




## Test accuracy
y_pred_test = logreg.predict(X_test)
y_pred_train = logreg.predict(X_train)

confusion_matrix(y_pred_test, y_test)
accuracy_score(y_pred_test, y_test)


print('Train Classification report:')
confusion_matrix(y_pred_train, y_train)
accuracy_score(y_pred_train, y_train)
print(classification_report(y_pred_train, y_train))
print(classification_report(y_pred_train, y_train))



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

#------------------------------------------------------

S = cats_mel_frequencies[3]
plt.figure(figsize=(9, 4))
librosa.display.specshow(librosa.power_to_db(S,ref=np.max),y_axis='mel', fmax=None,x_axis='time')
# plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()
