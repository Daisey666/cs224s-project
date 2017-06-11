import numpy as np
import os
import pdb
import random
import warnings

RAW_DATA_PATH = "/afs/ir/users/j/w/jwlouie/cs224s/final-project/cs224s-project/raw_data"
NUM_BUCKETS = 100
NUM_CEPSTRUMS = 13
NUM_SAMPLES = 4065
TRAIN_SPLIT = 0.8
TEST_SPLIT = 1 - TRAIN_SPLIT

def get_input_label_matrices(filenames):
    inputs = np.zeros((len(filenames), NUM_BUCKETS * NUM_CEPSTRUMS))
    labels = np.zeros((len(filenames)))
    i = 0
    for filename in filenames:
        raw_features = np.load(RAW_DATA_PATH + "/clip_mfccs/" + filename)
	raw_features = np.array_split(raw_features, NUM_BUCKETS)
	final_features = np.zeros((NUM_BUCKETS, NUM_CEPSTRUMS))
	final_features = np.array([np.nanmean(x, axis = 0) for x in raw_features])
        inputs[i] = np.ndarray.flatten(final_features)
	labels[i] = filename.split('_')[2] == "True.npy"
	i += 1
    return inputs, labels

def create_train_test_sets():
    filenames = os.listdir(RAW_DATA_PATH + "/clip_mfccs")    
    random.shuffle(filenames)
    train_set = filenames[ : int(NUM_SAMPLES * TRAIN_SPLIT)]
    test_set = filenames[int(NUM_SAMPLES * TRAIN_SPLIT) : ]
    train_inputs, train_labels = get_input_label_matrices(train_set)
    test_inputs, test_labels = get_input_label_matrices(test_set)
    pdb.set_trace()
    np.save(RAW_DATA_PATH + "/train_inputs.npy", train_inputs)
    np.save(RAW_DATA_PATH + "/train_labels.npy", train_labels)
    np.save(RAW_DATA_PATH + "/test_inputs.npy", test_inputs)
    np.save(RAW_DATA_PATH + "/test_labels.npy", test_labels)

#create_train_test_sets()

OPENSMILE_OUTFILE = "/afs/ir/users/j/w/jwlouie/cs224s/final-project/cs224s-project/raw_data/opensmile_features.arff"
OPENSMILE_DATASETS_PATH = "/afs/ir/users/j/w/jwlouie/cs224s/final-project/cs224s-project/raw_data/opensmile_datasets"

def create_train_test_opensmile():
    data = []
    with open(OPENSMILE_OUTFILE, "r") as f:
        for line in f: 
            if line[0] != "'":
                continue
            clip_data = line.strip().split(',')
            clip_name = clip_data[0]
            clip_features = np.array([float(x) for x in clip_data[1 : -1]])
            clip_label = int("True" in clip_name)
            data.append((clip_features, clip_label))
    random.shuffle(data)
    train_set = data[ : int(NUM_SAMPLES * TRAIN_SPLIT)]
    test_set = data[int(NUM_SAMPLES * TRAIN_SPLIT) : ]
    #pdb.set_trace()
    train_inputs = np.array([x[0] for x in train_set])
    train_labels = np.array([x[1] for x in train_set])
    test_inputs = np.array([x[0] for x in test_set])
    test_labels = np.array([x[1] for x in test_set])
    np.save(OPENSMILE_DATASETS_PATH + "/train_inputs.npy", train_inputs)
    np.save(OPENSMILE_DATASETS_PATH + "/train_labels.npy", train_labels)
    np.save(OPENSMILE_DATASETS_PATH + "/test_inputs.npy", test_inputs)
    np.save(OPENSMILE_DATASETS_PATH + "/test_labels.npy", test_labels)

# create_train_test_opensmile()

COMBINED_FEATURES = "/afs/ir/users/j/w/jwlouie/cs224s/final-project/cs224s-project/raw_data/features_100.txt"
COMBINED_LABELS = "/afs/ir/users/j/w/jwlouie/cs224s/final-project/cs224s-project/raw_data/labels_100.txt"
COMBINED_DATSETS_PATH = "/afs/ir/users/j/w/jwlouie/cs224s/final-project/cs224s-project/raw_data/combined_100"

def create_train_test_combined():
    data = []
    with open(COMBINED_FEATURES, "r") as f, open(COMBINED_LABELS, "r") as l:
        labels, features = l.readlines(), f.readlines()
        for label, feature in zip(labels, features): 
            clip_features = np.array(map(float, feature.strip().split()))
            clip_name = label 
            clip_label = int("True" in label)
            data.append((clip_features, clip_label))
    random.shuffle(data)
    train_set = data[ : int(NUM_SAMPLES * TRAIN_SPLIT)]
    test_set = data[int(NUM_SAMPLES * TRAIN_SPLIT) : ]
    #pdb.set_trace()
    train_inputs = np.array([x[0] for x in train_set])
    train_labels = np.array([x[1] for x in train_set])
    test_inputs = np.array([x[0] for x in test_set])
    test_labels = np.array([x[1] for x in test_set])
    np.save(COMBINED_DATSETS_PATH + "/train_inputs.npy", train_inputs)
    np.save(COMBINED_DATSETS_PATH + "/train_labels.npy", train_labels)
    np.save(COMBINED_DATSETS_PATH + "/test_inputs.npy", test_inputs)
    np.save(COMBINED_DATSETS_PATH + "/test_labels.npy", test_labels)

create_train_test_combined()


