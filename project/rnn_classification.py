import glob
import os
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
%matplotlib inline
plt.style.use('ggplot')

def windows(data, window_size):
    start = 0
    while start < len(data):
        yield int(start), int(start + window_size)
        start += (window_size / 2)

def extract_features(parent_dir,sub_dirs,file_ext="*.wav",bands = 20, frames = 41):
    window_size = 512 * (frames - 1)
    mfccs = []
    labels = []
    for l, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            sound_clip,s = librosa.load(fn)
            label = fn.split('/')[-1].split('_')[0]
            for (start,end) in windows(sound_clip,window_size):
                if(len(sound_clip[start:end]) == window_size):
                    signal = sound_clip[start:end]
                    mfcc = librosa.feature.mfcc(y=signal, sr=s, n_mfcc = bands).T.flatten()[:, np.newaxis].T
                    mfccs.append(mfcc)
                    labels.append(label)
    features = np.asarray(mfccs).reshape(len(mfccs),frames,bands)
    return np.array(features), np.array(labels,dtype = np.int)




bands = 20
frames = 20
window_size = 512 * (frames - 1)
files = glob.glob('./Cat-Dog/cats_dogs/*.wav')
fn = files[0]

parent_dir='Cat-Dog'
sub_dirs=['cats_dogs']

features, labels = extract_features(parent_dir, sub_dirs)


for (start,end) in windows(sound_clip,window_size):
    print(start, end)

start_stop = list(windows(sound_clip,window_size))

start, end = start_stop[0]


    if(len(sound_clip[start:end]) == window_size):
        signal = sound_clip[start:end]
        mfcc = librosa.feature.mfcc(y=signal, sr=s, n_mfcc = bands).T.flatten()[:, np.newaxis].T
        mfccs.append(mfcc)
        labels.append(label)

