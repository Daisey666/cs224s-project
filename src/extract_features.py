import numpy as np
import pdb
import soundfile as sf
import python_speech_features as psf
import os
import nltk_contrib.textgrid as tg

"""
Script to extract MFCC and other desired features from the recordings.
"""

ORIGINAL_DATA_PATH = "/afs/ir/users/j/w/jwlouie/cs224s/final-project/cs224s-project/CDC/data"
CLIPS_DATA_PATH = "/afs/ir/users/j/w/jwlouie/cs224s/final-project/cs224s-project/raw_data"
NUM_SUBJECTS = 32
SAMPLE_RATE = None

def split_audio():
    for subject_number in xrange(NUM_SUBJECTS):
        letter = chr(ord('A') + (subject_number % 4))        # A
        subj_id = "S-" + str(subject_number + 1) + letter    # S-1A
        audio_filename = ORIGINAL_DATA_PATH + "/" + subj_id + "/" + subj_id + "_R_16k.flac"
        split_filename = ORIGINAL_DATA_PATH + "/" + subj_id + "/" + subj_id + "_pedal_hand_corr.TextGrid"
        print split_filename
	data, SAMPLE_RATE = sf.read(audio_filename)
	textgrid_obj = tg.TextGrid.load(split_filename)
	for _, tier in enumerate(textgrid_obj):
	    lines = str(tier).split('\n')[1:]
	    for i in xrange(len(lines)):
	        clip_info = lines[i].strip().split()
		if len(clip_info) != 3 or clip_info[2] == "START":
		    continue
		start  = int(float(clip_info[0]) * SAMPLE_RATE)
 		end = int(float(clip_info[1]) * SAMPLE_RATE)
	        truth_value = clip_info[2] == "TRUTH"
		clip = np.array(data[start : end])
		total_len += len(clip)
		#np.save("%s/%d_%d_%r.npy" % (CLIPS_DATA_PATH, subject_number, i, truth_value), clip)
		clip_mfcc = psf.mfcc(clip, SAMPLE_RATE)
		#np.save("%s/%s/%d_%d_%r.npy" % (CLIPS_DATA_PATH, "clip_mfccs", subject_number, i, truth_value), clip_mfcc)

split_audio()
