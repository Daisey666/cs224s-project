import numpy as np
import os
import pdb
import random

RAW_DATA_PATH = "/afs/ir/users/j/w/jwlouie/cs224s/final-project/cs224s-project/raw_data"
NUM_BUCKETS = 100
NUM_CEPSTRUMS = 13
NUM_SAMPLES = 4100
TRAIN_SPLIT = 0.8
TEST_SPLIT = 1 - TRAIN_SPLIT

def get_input_label_matrices(filenames):
    inputs = np.zeros((len(filenames), NUM_BUCKETS * NUM_CEPSTRUMS))
    labels = np.zeros((len(filenames)))
    i = 0
    for filename in filenames: 
	mfcc_features = np.load(RAW_DATA_PATH + "/clip_mfccs/" + filename)
        final_features = np.array_split(mfcc_features, NUM_BUCKETS)
        final_features = np.array([np.mean(x, axis = 0) for x in final_features])
        inputs[i] = np.ndarray.flatten(final_features)
	labels[i] = '_'.split(filename) == "True"
	i += 1
    return inputs, labels

def create_train_test_sets():
    filenames = os.listdir(RAW_DATA_PATH + "/clip_mfccs")    
    random.shuffle(filenames)
    train_set = filenames[ : int(NUM_SAMPLES * TRAIN_SPLIT)]
    test_set = filenames[int(NUM_SAMPLES * TRAIN_SPLIT) : ]
    train_inputs, train_labels = get_input_label_matrices(train_set)
    test_inputs, test_labels = get_input_label_matrices(test_set)
    np.save(RAW_DATA_PATH + "/train_inputs.npy", train_inputs)
    np.save(RAW_DATA_PATH + "/train_labels.npy", train_labels)
    np.save(RAW_DATA_PATH + "/test_inputs.npy", test_inputs)
    np.save(RAW_DATA_PATH + "/test_labels.npy", test_labels)

create_train_test_sets()
