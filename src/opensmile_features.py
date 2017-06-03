
import sys
import pdb
import os
from os import listdir
from os.path import isfile, join
from subprocess import call

DATA_FILE_PATH = "/afs/ir/users/j/w/jwlouie/cs224s/final-project/cs224s-project/raw_data/wav_clips"
OUTPUT_FILE = "/afs/ir/users/j/w/jwlouie/cs224s/final-project/cs224s-project/raw_data/opensmile_features.arff"
OPENSMILE_PATH = "/afs/ir/users/j/w/jwlouie/cs224s/final-project/openSMILE-2.1.0/"
OPENSMILE_EXEC = "SMILExtract"

def extract_files(mypath):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    return onlyfiles


def main(args):
    print args
    filenames = extract_files(DATA_FILE_PATH)
    for filename in filenames:
        input_file = DATA_FILE_PATH + "/" + filename
        filename = filename.split(".")[0]
        os.system(' '.join([OPENSMILE_PATH + OPENSMILE_EXEC, "-C", OPENSMILE_PATH +  "/config/IS09_emotion.conf", "-N", filename, "-I", input_file, "-O", OUTPUT_FILE]))

main(sys.argv[1:])
