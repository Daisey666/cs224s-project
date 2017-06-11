import numpy as np
import os
import sys
import pdb
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.combine import SMOTEENN

COMBINED_FEATURES = "/afs/ir/users/j/w/jwlouie/cs224s/final-project/cs224s-project/raw_data/features_100_stripped.txt"
COMBINED_LABELS = "/afs/ir/users/j/w/jwlouie/cs224s/final-project/cs224s-project/raw_data/labels_100_stripped.txt"
NUM_FEATURES = 1000000
OVER_SAMPLE = True

def get_data(data):
    features, labels = [], []
    selected_data = COMBINED_FEATURES
    selected_labels = COMBINED_LABELS
    if "50" in data:
        selected_data = "/afs/ir/users/j/w/jwlouie/cs224s/final-project/cs224s-project/raw_data/features_50_stripped.txt"
        selected_labels = "/afs/ir/users/j/w/jwlouie/cs224s/final-project/cs224s-project/raw_data/labels_50_stripped.txt"
    elif "200" in data:
        selected_data = "/afs/ir/users/j/w/jwlouie/cs224s/final-project/cs224s-project/raw_data/features_200_stripped.txt"
        selected_labels = "/afs/ir/users/j/w/jwlouie/cs224s/final-project/cs224s-project/raw_data/labels_200_stripped.txt"
    elif "300" in data:
        selected_data = "/afs/ir/users/j/w/jwlouie/cs224s/final-project/cs224s-project/raw_data/features_300_stripped.txt"
        selected_labels = "/afs/ir/users/j/w/jwlouie/cs224s/final-project/cs224s-project/raw_data/labels_300_stripped.txt"
    with open(selected_data, "r") as f, open(selected_labels, "r") as l:
        labels_str, features_str = l.readlines(), f.readlines()
        for label, feature in zip(labels_str, features_str): 
            clip_features = np.array(map(float, feature.strip().split())[:NUM_FEATURES])
            clip_name = label 
            clip_label = int("True" in label)
            features.append(clip_features)
            labels.append(clip_label)
    #pdb.set_trace()
    X, y = np.array(features), np.array(labels)
    if OVER_SAMPLE:
        #sm = SMOTE(random_state=1)
        #sm = RandomOverSampler(random_state=1)
        sm = SMOTEENN(random_state=1)
        X, y = sm.fit_sample(X, y)
    print "num of one class: %d, total: %d" % (sum(y), len(y))
    return X, y

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                                n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5), selected_model=""):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig("learning_curve_" + selected_model + ".png")
    print "learning curve saved!"
    return plt

def perform_kfold_cv(args):
    folds = int(args[0])
    data, labels = get_data(args[1])

    # Try some feature selection:
    # Variance threshold:
    selector = VarianceThreshold(threshold=.9)#threshold=.8 * (1 - .8))
    data = selector.fit_transform(data)
    print data.shape

    selected_model = args[2]
    if "nn" in selected_model:
        title = "Learning Curves (MLP)"
        model = MLPClassifier(solver='lbfgs', learning_rate='adaptive', alpha=100, random_state=1)
    elif "log" in selected_model:
        title = "Learning Curves (Logistic Regression)"
        model = linear_model.LogisticRegression(C=.00001)
    elif "svc" in selected_model:
        title = "Learning Curves (Support Vector Classifier)"
        model = svm.SVC(C=.00001) # change to be other models if necessary
    elif "boost" in selected_model:
        title = "Learning Curves (Gradient Boosting Classifier)"
        model = GradientBoostingClassifier(random_state=1)
    elif "bag" in selected_model:
        title = "Learning Curves (Bagging Classifier)"
        model = BaggingClassifier(random_state=1)

    scores = cross_val_score(model, data, labels, cv=folds)
    print("Accuracy: %0.6f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # Compute average F1 score:

    average_f1 = 0.0
    for rstate in xrange(folds):
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=1.0 / float(folds), random_state=rstate)
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        average_f1 += f1_score(y_test, predicted)
    average_f1 /= float(folds)
    print("F1 Score: %0.6f" % (average_f1))

    print("Baseline F1 Score: %0.6f" % (f1_score(labels, [1 for i in xrange(len(labels))])))

    plot_learning_curve(model, title, data, labels, ylim=(0.0, 1.01), cv=ShuffleSplit(n_splits=10, test_size=1.0 / float(folds), random_state=0), n_jobs=4, selected_model=selected_model)
    #plt.show()

perform_kfold_cv(sys.argv[1:])

def test():
    t = np.arange(0.0, 2.0, 0.01)
    s = 1 + np.sin(2*np.pi*t)
    plt.plot(t, s)

    plt.xlabel('time (s)')
    plt.ylabel('voltage (mV)')
    plt.title('About as simple as it gets, folks')
    plt.grid(True)
    plt.savefig("test.png")
    plt.show()

#test()
