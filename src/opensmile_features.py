
import sys
import pdb
import os
from os import listdir
from os.path import isfile, join
from subprocess import call

DATA_FILE_PATH = "/afs/ir/users/j/w/jwlouie/cs224s/final-project/cs224s-project/raw_data/wav_clips_stripped"
OUTPUT_FILE = "/afs/ir/users/j/w/jwlouie/cs224s/final-project/cs224s-project/raw_data/opensmile_features_stripped.arff"
OPENSMILE_PATH = "/afs/ir/users/j/w/jwlouie/cs224s/final-project/openSMILE-2.1.0/"
OPENSMILE_EXEC = "SMILExtract"

def extract_files(mypath):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    return onlyfiles


def main(args):
    print args
    filenames = extract_files(DATA_FILE_PATH)
    counter = 0
    for filename in filenames:
        # use 15 and 31 because of zero indexing (we want to not use subjects 16 or 32 in reality)
        if filename[0:len("15_")] == "15_" or filename[0:len("31_")] == "31_": continue
        input_file = DATA_FILE_PATH + "/" + filename
        filename = filename.split(".")[0]
        counter += 1
        os.system(' '.join([OPENSMILE_PATH + OPENSMILE_EXEC, "-C", OPENSMILE_PATH +  "/config/IS09_emotion.conf", "-N", filename, "-I", input_file, "-O", OUTPUT_FILE]))
    print counter

main(sys.argv[1:])
