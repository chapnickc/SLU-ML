import glob
import os
import librosa
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import numpy as np

from tqdm import tqdm

%pylab
import matplotlib.pyplot as plt
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
        for fn in tqdm(glob.glob(os.path.join(parent_dir, sub_dir, file_ext))):
            sound_clip,s = librosa.load(fn)
            label = fn.split('/')[-1].split('_')[0]
            for (start,end) in windows(sound_clip,window_size):
                if(len(sound_clip[start:end]) == window_size):
                    signal = sound_clip[start:end]
                    mfcc = librosa.feature.mfcc(y=signal, sr=s, n_mfcc = bands).T.flatten()[:, np.newaxis].T
                    mfccs.append(mfcc)
                    labels.append(label)
    features = np.asarray(mfccs).reshape(len(mfccs),frames,bands)
    return np.array(features), np.array(labels)

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode





#-----------------------------------------
# bands = 20
# frames = 20
# window_size = 512 * (frames - 1)
# files = glob.glob('./Cat-Dog/cats_dogs/*.wav')
# fn = files[0]

parent_dir='Cat-Dog'
sub_dirs=['cats_dogs']

features, labels = extract_features(parent_dir, sub_dirs)

enc = LabelBinarizer(sparse_output=False)
y =enc.fit_transform(labels)






X_train, X_test, y_train, y_test = train_test_split(features, y,
                                        test_size=0.33,
                                        random_state=42
                                    )



#---------------------------------------------------

# parent_dir=['Cat-Dog']
# sub_dirs = ['cats_dogs']
# tr_sub_dirs = ['cats_dogs']
# tr_features,tr_labels = extract_features(parent_dir,tr_sub_dirs)
# tr_labels = one_hot_encode(tr_labels)

# ts_sub_dirs = ['fold2']
# ts_features,ts_labels = extract_features(parent_dir,ts_sub_dirs)
# ts_labels = one_hot_encode(ts_labels)

#-----------------------------------------

tf.reset_default_graph()


learning_rate = 0.01
training_iters = 1000
batch_size = 50
display_step = 200

# Network Parameters
n_input = 20
n_steps = 41
n_hidden = 300
n_classes = 10

x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

weight = tf.Variable(tf.random_normal([n_hidden, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))


def RNN(x, weight, bias):
    cell = rnn_cell.LSTMCell(n_hidden,state_is_tuple = True)
    cell = rnn_cell.MultiRNNCell([cell] * 2)
    output, state = tf.nn.dynamic_rnn(cell, x, dtype = tf.float32)
    output = tf.transpose(output, [1, 0, 2])
    last = tf.gather(output, int(output.get_shape()[0]) - 1)
    return tf.nn.softmax(tf.matmul(last, weight) + bias)


prediction = RNN(x, weight, bias)

# Define loss and optimizer
loss_f = -tf.reduce_sum(y * tf.log(prediction))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss_f)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()


with tf.Session() as session:
    session.run(init)

    for itr in range(training_iters):
        offset = (itr * batch_size) % (y_train.shape[0] - batch_size)
        batch_x = X_train[offset:(offset + batch_size), :, :]
        batch_y = y_train[offset:(offset + batch_size), :]
        _, c = session.run([optimizer, loss_f],feed_dict={x: batch_x, y : batch_y})

        if epoch % display_step == 0:
            # Calculate batch accuracy
            acc = session.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = session.run(loss_f, feed_dict={x: batch_x, y: batch_y})
            print "Iter " + str(epoch) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc)

    print('Test accuracy: ',round(session.run(accuracy, feed_dict={x: X_test, y: y_test}) , 3))
