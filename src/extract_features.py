import numpy as np
import pdb
import soundfile as sf
import python_speech_features as psf
import os
import os.path
import nltk_contrib.textgrid as tg
import scikits.audiolab

"""
Script to extract MFCC and other desired features from the recordings.
"""

ORIGINAL_DATA_PATH = "/afs/ir/users/j/w/jwlouie/cs224s/final-project/cs224s-project/CDC/data"
CLIPS_DATA_PATH = "/afs/ir/users/j/w/jwlouie/cs224s/final-project/cs224s-project/raw_data"
NUM_SUBJECTS = 32
SAMPLE_RATE = None

def read_file_lines(f):
    lines = f.readlines()
    lines = [ _.strip().split() for _ in lines]
    # parse out (time, word) 
    time_and_word = [(float(line[2]), line[4].lower()) for line in lines]
    return time_and_word

def sanity_check():
    counter = 0
    print "sanity check"
    for subject_number in xrange(NUM_SUBJECTS):
        letter = chr(ord('A') + (subject_number % 4))        # A
        subj_id = "S-" + str(subject_number + 1) + letter    # S-1A
        if "32" in subj_id  or "16" in subj_id: continue
        print subj_id
        audio_filename = ORIGINAL_DATA_PATH + "/" + subj_id + "/" + subj_id + "_R_16k.flac"
        split_filename = ORIGINAL_DATA_PATH + "/" + subj_id + "/" + subj_id + "_pedal_hand_corr.TextGrid"
        data, SAMPLE_RATE = sf.read(audio_filename)
        textgrid_obj = tg.TextGrid.load(split_filename)
        for _, tier in enumerate(textgrid_obj):
            lines = str(tier).split('\n')[1:]
            for i in xrange(len(lines)):
                
                # skip clips that are unlabelled
                clip_info = lines[i].strip().split()
                if len(clip_info) != 3 or clip_info[2] == "START":
                    continue
                counter += 1
    print "clip count: " + str(counter)

    counter = 0
    for subject_number in xrange(NUM_SUBJECTS):
        letter = chr(ord('A') + (subject_number % 4))        # A
        subj_id = "S-" + str(subject_number + 1) + letter    # S-1A
        interviewer_filename = ORIGINAL_DATA_PATH + "/" + subj_id + "/" + subj_id + "_L_16k-wrd.ctm"
        interviewee_filename = ORIGINAL_DATA_PATH + "/" + subj_id + "/" + subj_id + "_R_16k-wrd.ctm"
        split_filename = ORIGINAL_DATA_PATH + "/" + subj_id + "/" + subj_id + "_pedal_hand_corr.TextGrid"
        textgrid_obj = tg.TextGrid.load(split_filename)
        if not os.path.isfile(interviewer_filename) or not os.path.isfile(interviewee_filename): continue
        print subj_id
        with open(interviewer_filename, "r") as interviewer, open(interviewee_filename, "r") as interviewee:
            interviewer_time_word, interviewee_time_word = read_file_lines(interviewer), read_file_lines(interviewee)
            for _, tier in enumerate(textgrid_obj):
                lines = str(tier).split('\n')[1:]
                for i in xrange(len(lines)):
                    clip_info = lines[i].strip().split()
                    if len(clip_info) != 3 or clip_info[2] == "START":
                        continue
                    interviewer_time_word, interviewee_time_word = read_file_lines(interviewer), read_file_lines(interviewee)
                    interviewer_to_write, interviewee_to_write = [], []
                    start, end  = float(clip_info[0]), float(clip_info[1])
                    truth_value = clip_info[2] == "TRUTH"
                    # Read from the interviewer first:
                    while len(interviewer_time_word) > 0:
                        time, word = interviewer_time_word[0]
                        if time <= end and  time >= start:
                            interviewer_to_write.append(word)
                        elif time > end: break
                        interviewer_time_word = interviewer_time_word[1:]

                    # Read from the interviewee now:
                    while len(interviewee_time_word) > 0:
                        time, word = interviewee_time_word[0]
                        if time <= end and  time >= start:
                            interviewee_to_write.append(word)
                        elif time > end: break
                        interviewee_time_word = interviewee_time_word[1:]
                    counter += 1

    print "sentence count: " + str(counter)

sanity_check()

def extract_words():
    output_filename = CLIPS_DATA_PATH + "/sentences.txt"
    with open(output_filename, "w+") as output:
        for subject_number in xrange(NUM_SUBJECTS):
            letter = chr(ord('A') + (subject_number % 4))        # A
            subj_id = "S-" + str(subject_number + 1) + letter    # S-1A
            interviewer_filename = ORIGINAL_DATA_PATH + "/" + subj_id + "/" + subj_id + "_L_16k-wrd.ctm"
            interviewee_filename = ORIGINAL_DATA_PATH + "/" + subj_id + "/" + subj_id + "_R_16k-wrd.ctm"
            split_filename = ORIGINAL_DATA_PATH + "/" + subj_id + "/" + subj_id + "_pedal_hand_corr.TextGrid"
            textgrid_obj = tg.TextGrid.load(split_filename)
            if not os.path.isfile(interviewer_filename) or not os.path.isfile(interviewee_filename): continue
            with open(interviewer_filename, "r") as interviewer, open(interviewee_filename, "r") as interviewee:
                output.write(subj_id + "\n")
                interviewer_time_word, interviewee_time_word = read_file_lines(interviewer), read_file_lines(interviewee)
                for _, tier in enumerate(textgrid_obj):
                    lines = str(tier).split('\n')[1:]
                    for i in xrange(len(lines)):
                        clip_info = lines[i].strip().split()
                        if len(clip_info) != 3 or clip_info[2] == "START":
                            continue
                        interviewer_to_write, interviewee_to_write = [], []
                        start, end  = float(clip_info[0]), float(clip_info[1])
                        truth_value = clip_info[2] == "TRUTH"
                        # Read from the interviewer first:
                        while len(interviewer_time_word) > 0:
                            time, word = interviewer_time_word[0]
                            if time <= end and  time >= start:
                                interviewer_to_write.append(word)
                            elif time > end: break
                            interviewer_time_word = interviewer_time_word[1:]
                        output.write("question: " + " ".join(interviewer_to_write) + "\n")

                        # Read from the interviewee now:
                        while len(interviewee_time_word) > 0:
                            time, word = interviewee_time_word[0]
                            if time <= end and  time >= start:
                                interviewee_to_write.append(word)
                            elif time > end: break
                            interviewee_time_word = interviewee_time_word[1:]
                        output.write("answer: " + " ".join(interviewee_to_write) + "\n\n")

        #pdb.set_trace()
                    


# extract_words()

def split_audio():
    """
    Splits each file into clips 
    """
    for subject_number in xrange(NUM_SUBJECTS):
        letter = chr(ord('A') + (subject_number % 4))        # A
        subj_id = "S-" + str(subject_number + 1) + letter    # S-1A
        audio_filename = ORIGINAL_DATA_PATH + "/" + subj_id + "/" + subj_id + "_R_16k.flac"
        split_filename = ORIGINAL_DATA_PATH + "/" + subj_id + "/" + subj_id + "_pedal_hand_corr.TextGrid"
        data, SAMPLE_RATE = sf.read(audio_filename)
        textgrid_obj = tg.TextGrid.load(split_filename)
        for _, tier in enumerate(textgrid_obj):
            lines = str(tier).split('\n')[1:]
            for i in xrange(len(lines)):
                
                # skip clips that are unlabelled
                clip_info = lines[i].strip().split()
                if len(clip_info) != 3 or clip_info[2] == "START":
                    continue
                start  = int(float(clip_info[0]) * SAMPLE_RATE)
                end = int(float(clip_info[1]) * SAMPLE_RATE)
                truth_value = clip_info[2] == "TRUTH"
                clip = np.array(data[start : end])

                # save clip as wav
                wav_filename = "%s/%s/%d_%d_%r.wav" % (CLIPS_DATA_PATH, "wav_clips", subject_number, i, truth_value)
                scikits.audiolab.wavwrite(clip, wav_filename, fs=SAMPLE_RATE, enc='pcm32')

                # extract mfcc features and write to .npy
                clip_mfcc = psf.mfcc(clip, SAMPLE_RATE)
                if clip_mfcc.shape[0] >= 100:       
                    np.save("%s/%s/%d_%d_%r.npy" % (CLIPS_DATA_PATH, "clip_mfccs", subject_number, i, truth_value), clip_mfcc)

#split_audio()
