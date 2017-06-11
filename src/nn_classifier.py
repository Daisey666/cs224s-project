
import numpy as np
import pdb
from sklearn.neural_network import MLPClassifier

COMBINED_PATH = "/afs/ir/users/j/w/jwlouie/cs224s/final-project/cs224s-project/raw_data/combined_100/"

PATH = COMBINED_PATH 
TRAIN_INPUTS = PATH + "train_inputs.npy"
TRAIN_LABELS = PATH + "train_labels.npy"
TEST_INPUTS = PATH + "test_inputs.npy"
TEST_LABELS = PATH + "test_labels.npy"

def read_dataset(inputs, labels):
    return np.load(inputs), np.load(labels)

X_train, y_train = read_dataset(TRAIN_INPUTS, TRAIN_LABELS)
X_test, y_test = read_dataset(TEST_INPUTS, TEST_LABELS)

clf = MLPClassifier(solver='lbfgs', learning_rate='adaptive', alpha=1e-5, random_state=1)

clf.fit(X_train, y_train)

score = clf.score(X_test, y_test)

print score

pdb.set_trace()
