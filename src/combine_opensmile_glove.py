
import pdb

GLOVE_FILENAME = "/afs/ir.stanford.edu/users/j/w/jwlouie/cs224s/final-project/cs224s-project/raw_data/qa_glove_features_50.txt"
OPENSMILE_FILENAME = "/afs/ir.stanford.edu/users/j/w/jwlouie/cs224s/final-project/cs224s-project/raw_data/opensmile_features_stripped.arff"
OUTPUT_FILENAME = "/afs/ir.stanford.edu/users/j/w/jwlouie/cs224s/final-project/cs224s-project/raw_data/features_50_stripped.txt"
LABELS_FILENAME = "/afs/ir.stanford.edu/users/j/w/jwlouie/cs224s/final-project/cs224s-project/raw_data/labels_50_stripped.txt"

def combine_features():
    labels = []
    glove_features = []
    with open(GLOVE_FILENAME, "r") as f:
        lines = f.readlines()
        glove_features = [map(float, line.strip().split()) for line in lines]

    print len(glove_features), len(glove_features[0])

    opensmile_features = []
    with open(OPENSMILE_FILENAME, "r") as f:
        lines = f.readlines()[391:]
        labels = [line.strip().split(",")[0] for line in lines]
        opensmile_features = [map(float, line.strip().split(",")[1:-1]) for line in lines]

    print len(opensmile_features), len(opensmile_features[0])

    with open(LABELS_FILENAME, "w+") as f:
        for label in labels:
            f.write(label + "\n")

    print "labels written"

    with open(OUTPUT_FILENAME, "w+") as f:
        for opensmile_feature, glove_feature in zip(opensmile_features, glove_features):
            combined_features = map(str, opensmile_feature + glove_feature)
            f.write(" ".join(combined_features) + "\n")


    print "features written"

    #pdb.set_trace()



combine_features()
