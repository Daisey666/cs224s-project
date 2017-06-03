import numpy as np
import matplotlib.pyplot as plt
import pdb

from sklearn import linear_model

PATH = "/afs/ir/users/j/w/jwlouie/cs224s/final-project/cs224s-project/raw_data/"
TRAIN_INPUTS = PATH + "train_inputs.npy"
TRAIN_LABELS = PATH + "train_labels.npy"
TEST_INPUTS = PATH + "test_inputs.npy"
TEST_LABELS = PATH + "test_labels.npy"

def read_dataset(inputs, labels):
    return np.load(inputs), np.load(labels)

X_train, y_train = read_dataset(TRAIN_INPUTS, TRAIN_LABELS)
X_test, y_test = read_dataset(TEST_INPUTS, TEST_LABELS)

pdb.set_trace()
logreg = linear_model.LogisticRegression()
logreg.fit(X_train, y_train)
score = logreg.score(X_test, y_test)
print score

