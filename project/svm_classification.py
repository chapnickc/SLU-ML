from utils import parse_audio_files

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
%pylab


features, labels = parse_audio_files(
                        parent_dir='Cat-Dog',
                        sub_dirs=['cats_dogs']
                    )
npz = np.load('checkpoint/features_and_labels.npz')
features = npz['arr_0']
labels = npz['arr_1']



X_train, X_test, y_train, y_test = train_test_split(
                                        features,
                                        labels,
                                        test_size=0.3,
                                        random_state=42
                                    )


#--------------------------------
# Linear Kernel
#--------------------------------
clf = svm.SVC(kernel='linear', C=1, random_state=42)
clf.fit(X_train, y_train)


# Test Classification Report
y_pred_test = clf.predict(X_test)
print( confusion_matrix(y_pred_test, y_test) )
print( accuracy_score(y_pred_test, y_test) )
print( classification_report(y_pred_test, y_test) )

p,r,f,s = precision_recall_fscore_support(y_pred_test, y_test, average='micro')

print("F-Score:", round(f,3))

# Train Classification Report
y_pred_train = clf.predict(X_train)
print( confusion_matrix(y_pred_train, y_train) )
print( accuracy_score(y_pred_train, y_train) )
print( classification_report(y_pred_train, y_train) )



#--------------------------------
# Degree 2 Polynomial Kernel
#--------------------------------
clf = svm.SVC(kernel='poly', C=1, degree=2, random_state=42)
clf.fit(X_train, y_train)


# Test Classification Report
y_pred_test = clf.predict(X_test)
print( confusion_matrix(y_pred_test, y_test) )
print( accuracy_score(y_pred_test, y_test) )
print( classification_report(y_pred_test, y_test) )

p,r,f,s = precision_recall_fscore_support(y_pred_test, y_test, average='micro')

print("F-Score:", round(f,3))

# Train Classification Report
y_pred_train = clf.predict(X_train)
print( confusion_matrix(y_pred_train, y_train) )
print( accuracy_score(y_pred_train, y_train) )
print( classification_report(y_pred_train, y_train) )




#--------------------------------
# RBF Kernel
#--------------------------------

C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
grid.fit(features, labels)

scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
                                                     len(gamma_range))


# Utility function to move the midpoint of a colormap to be around
# the values of interest.

class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

# Draw heatmap of the validation accuracy as a function of gamma and C
#
# The score are encoded as colors with the hot colormap which varies from dark
# red to bright yellow. As the most interesting scores are all located in the
# 0.92 to 0.97 range we use a custom normalizer to set the mid-point to 0.92 so
# as to make it easier to visualize the small variations of score values in the
# interesting range while not brutally collapsing all the low score values to
# the same color.

plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
           norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
plt.yticks(np.arange(len(C_range)), C_range)
plt.title('Validation accuracy')
plt.show()


clf = svm.SVC(kernel='rbf', C=100, gamma=1e-6, random_state=42)
clf.fit(X_train, y_train)


# Test Classification Report
y_pred_test = clf.predict(X_test)
print( confusion_matrix(y_pred_test, y_test) )
print( accuracy_score(y_pred_test, y_test) )
print( classification_report(y_pred_test, y_test) )

p,r,f,s = precision_recall_fscore_support(y_pred_test, y_test, average='micro')

print("F-Score:", round(f,3))

# Train Classification Report
y_pred_train = clf.predict(X_train)
print( confusion_matrix(y_pred_train, y_train) )
print( accuracy_score(y_pred_train, y_train) )
print( classification_report(y_pred_train, y_train) )


