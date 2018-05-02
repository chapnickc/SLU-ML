from utils import one_hot_encode

import glob
import os
import librosa
import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support

import matplotlib.pyplot as plt

from tqdm import tqdm

def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
    return mfccs,chroma,mel,contrast,tonnetz

def parse_audio_files(parent_dir,sub_dirs,file_ext='*.wav'):
    features, labels = np.empty((0,193)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        for fn in tqdm(glob.glob(os.path.join(parent_dir, sub_dir, file_ext))):
            mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
            ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            features = np.vstack([features,ext_features])
            labels = np.append(labels, fn.split('/')[-1].split('_')[0])
    return np.array(features), np.array(labels)


parent_dir='Cat-Dog'
sub_dirs=['cats_dogs']

features, labels = parse_audio_files(parent_dir,sub_dirs)

# np.savez('features_and_labels.npz', features, labels)

npz = np.load('checkpoint/features_and_labels.npz')
features = npz['arr_0']
labels = npz['arr_1']

labels = pd.factorize(labels)

# labels[0] is lables labels[1] is uniques
labels = one_hot_encode(labels[0])

train_test_split = np.random.rand(len(features)) < 0.70
train_x = features[train_test_split]
train_y = labels[train_test_split]
test_x = features[~train_test_split]
test_y = labels[~train_test_split]


training_epochs = 5000
n_dim = features.shape[1]
n_classes = len(labels[1])
n_hidden_units_one = 280
n_hidden_units_two = 300
sd = 1 / np.sqrt(n_dim)
learning_rate = 0.01


kwargs={'mean': 0, 'stddev': sd}
X = tf.placeholder(tf.float32,[None, n_dim])
Y = tf.placeholder(tf.float32,[None, n_classes])

W_1 = tf.Variable(tf.random_normal([n_dim, n_hidden_units_one], **kwargs))
b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], **kwargs))
h_1 = tf.nn.tanh(tf.matmul(X, W_1) + b_1)


W_2 = tf.Variable(tf.random_normal([n_hidden_units_one,n_hidden_units_two], **kwargs))
b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], **kwargs))
h_2 = tf.nn.sigmoid(tf.matmul(h_1, W_2) + b_2)


W = tf.Variable(tf.random_normal([n_hidden_units_two,n_classes], **kwargs))
b = tf.Variable(tf.random_normal([n_classes], **kwargs))
y_ = tf.nn.softmax(tf.matmul(h_2, W) + b)

init = tf.global_variables_initializer()

cost_function = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y_), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

cost_history = np.empty(shape=[1],dtype=float)
y_true, y_pred = None, None

with tf.Session() as sess:
    sess.run(init)
    for epoch in tqdm(range(training_epochs)):
        _, cost = sess.run([optimizer,cost_function],feed_dict={X:train_x,Y:train_y})
        cost_history = np.append(cost_history,cost)
    y_pred = sess.run(tf.argmax(y_,1),feed_dict={X: test_x})
    y_true = sess.run(tf.argmax(test_y,1))



test_accuracy = np.sum(np.array([y_true == y_pred]))/len(y_true)
print(f'Test Accuracy: {test_accuracy}')


fig = plt.figure(figsize=(10,8))
plt.plot(cost_history)
plt.ylabel("Cost")
plt.xlabel("Iterations")
plt.axis([0,training_epochs,0,np.max(cost_history)])
plt.show()

p,r,f,s = precision_recall_fscore_support(y_true, y_pred, average='micro')
print("F-Score:", round(f,3))

print(classification_report(y_true, y_pred))
