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


def get_file_num(f):
    fpath, ext = os.path.splitext(f)
    return int(fpath.split('_')[-1])


def load_data(audio_files, sr=16000):
    samples = []
    channels = []
    for filename in audio_files:
        samples.append( librosa.load(filename, sr=sr) )
        with audioread.audio_open(filename) as input_file:
            channels.append(input_file.channels)

    channels = list(set(channels))
    if len(channels)==1 and channels[0]== 1:
        print("Single Channel Audio")
    else:
        warnings.warn("Some audio files have several channels", UserWarning)

    return samples


def compute_mel_frequencies(audio_list, sr=16000):
    mel_spectrograms = []
    for i in range(len(audio_list)):
        spec = librosa.feature.melspectrogram(y=np.array(audio_list[i][0]), sr=sr)
        mel_spectrograms.append(spec)
    return mel_spectrograms


def compute_mfccs(mel_spectrograms, sr=16000):
    mfccs = []
    for spec in mel_spectrograms:
        mfccs.append( librosa.feature.mfcc(S=librosa.power_to_db(spec), sr=sr) )
    return mfccs


def compute_deltas(mfccs_list):
    deltas = []
    for mfccs in mfccs_list:
        deltas.append(librosa.feature.delta(mfccs))
    return deltas


def remove_frames(audio_list, frames_list, sr=16000):
    """
    Collect frames from the MFCCS which correspond
    to the estimated positions of detected onsets.

    Returns:
        A list of reduced MFCC frames, each element
        in the list has a shape of
        (number of mfccs)x(number of detected onsets)

    ..note: The number of detected onsets varies in each sample
    """
    detect = []
    for sample in audio_list:
        y, sr = sample
        detect.append(librosa.onset.onset_detect(y=y))

    new_frames = []
    for k in range(len(frames_list)):
        cleaned = [
                [ frames_list[k][i][j] for j in detect[k] ]
                for i in range(frames_list[0].shape[0])
            ]
        new_frames.append(cleaned)

    return new_frames


def build_df_mfccs(mfccs_list, label=None):
    """
    Convert the MFCCS of each onset detection into an
    example to be used in training.

    Args:
        mfccs_list: a list of mfcc arrays each of size
                    (number of MFCCS)x(number of detections)
    Returns:
        ...
    """
    # columns are the mfccs bins (ie. features)
    # index corresponds to the sample that it came from
    df = pd.DataFrame()
    for i in range(len(mfccs_list)):
        df_temp = pd.DataFrame(
                np.transpose(mfccs_list[i]),
                index=np.ones(len(np.transpose(mfccs_list[i])))*(i+1)
            )
        frames = [df, df_temp]
        df = pd.concat(frames)
    df['Label']=label
    return df


def split_data(cats_frames, dogs_frames, n_cats, n_dogs, test_size=0.3):
    y_cats = np.zeros(n_cats)
    y_dogs = np.ones(n_dogs)

    #  train test splits for cats and dogs
    X_train_cats, X_test_cats, y_train_cats, y_test_cats = \
            train_test_split(cats_frames, y_cats, test_size=test_size, random_state=42)
    X_train_dogs, X_test_dogs, y_train_dogs, y_test_dogs = \
            train_test_split(dogs_frames, y_dogs, test_size=test_size, random_state=42)

    # build the training dataframe
    df_train_cats = build_df_mfccs(X_train_cats, label=0)
    df_train_dogs = build_df_mfccs(X_train_dogs, label=1)
    result = pd.concat([df_train_cats, df_train_dogs])
    df_train = result.sample(frac=1, random_state=42)

    # build the test dataframe
    df_test_cats = build_df_mfccs(X_test_cats, label=0)
    df_test_dogs = build_df_mfccs(X_test_dogs, label=1)
    result = pd.concat([df_test_cats, df_test_dogs])
    df_test = result.sample(frac=1, random_state=42)
    return df_train, df_test











def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode




def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz
