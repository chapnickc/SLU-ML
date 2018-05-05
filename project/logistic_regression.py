from utils import parse_audio_files

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support

features, labels = parse_audio_files(parent_dir='Cat-Dog', sub_dirs=['cats_dogs'])

X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                        test_size=0.3,
                                        random_state=42
                                    )

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Test Classification Report
y_pred_test = logreg.predict(X_test)
print(confusion_matrix(y_pred_test, y_test))
print(accuracy_score(y_pred_test, y_test))
print(classification_report(y_pred_test, y_test))

p,r,f,s = precision_recall_fscore_support(y_pred_test, y_test, average='micro')
print("F-Score:", round(f,3))

# Train Classification Report
y_pred_train = logreg.predict(X_train)
print(confusion_matrix(y_pred_train, y_train))
print(accuracy_score(y_pred_train, y_train))
print(classification_report(y_pred_train, y_train))


