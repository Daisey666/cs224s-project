import numpy as np
import pdb
import soundfile as sf
import python_speech_features as psf
import os

"""
Script to extract MFCC and other desired features from the recordings.
"""

DATA_PATH = "/afs/ir/users/j/w/jwlouie/cs224s/final-project/cs224s-project/CDC/data/"
NUM_SUBJECTS = 32

def split_audio():
    for i in xrange(NUM_SUBJECTS):
        letter = chr(ord('A') + (i % 4))        # A
        subj_id = "S-" + str(i + 1) + letter    # S-1A
        audio_filename = DATA_PATH + subj_id + "/" + subj_id + "_R_16k.flac"
        split_filename = DATA_PATH + subj_id + "/" + subj_id + "_pedal_hand_corr.TextGrid"
        data, sample_rate = sf.read(audio_filename)
        #mfcc_features = psf.mfcc(data, sample_rate)
        with open(split_filename) as splitfile:
            pass
            # TODO: use NLTK TextGrid parser to split data into different arrays
            # save as <subject number> <clip number> <truth value> 
        
        pdb.set_trace()


split_audio()
