import numpy as np
import pdb
import soundfile as sf
import python_speech_features as psf
import os
import os.path
import nltk_contrib.textgrid as tg
import scikits.audiolab
from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import audioSegmentation as aS

"""
Contains functions to: 
    - split the audio into labelled clips
    - extract MFCC features from the recordings
"""

HOMEDIR = "/afs/ir/users/j/w/jwlouie/cs224s/final-project/cs224s-project"
ORIGINAL_DATA_PATH = HOMEDIR + "/CDC/data"
PROCESSED_DATA_PATH = HOMEDIR + "/raw_data"
WAV_CLIPS_PATH = PROCESSED_DATA_PATH + "/wav_clips_stripped"
CLIP_MFCCS_PATH = PROCESSED_DATA_PATH + "/clip_mfccs"
RNN_INPUTS_PATH = PROCESSED_DATA_PATH + "/rnn_inputs"
RNN_INPUTS_ARFF_FILE = PROCESSED_DATA_PATH + "/rnn_inputs.arff"

OPENSMILE_PATH = "/afs/ir/users/j/w/jwlouie/cs224s/final-project/openSMILE-2.1.0"
OPENSMILE_EXEC = OPENSMILE_PATH + "/SMILExtract"
OPENSMILE_CONFIG = OPENSMILE_PATH + "/config/IS09_emotion.conf"

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
        audio_filename = "%s/%s/%s_R_16k.flac" % (ORIGINAL_DATA_PATH, subj_id, subj_id)
        split_filename = "%s/%s/%s_pedal_hand_corr.TextGrid" % \
                                (ORIGINAL_DATA_PATH, subj_id, subj_id)
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

def extract_words():
    output_filename = PROCESSED_DATA_PATH + "/sentences.txt"
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

def split_audio():
    """
    Splits each file into clips, saved into WAV_CLIPS_PATH
    Also extracts mfcc features of each clip, saved into CLIP_MFCCS_PATH
    """
    for subject_number in xrange(NUM_SUBJECTS):
        if subject_number == 15 or subject_number == 31: continue
        letter = chr(ord('A') + (subject_number % 4))        # A
        subj_id = "S-" + str(subject_number + 1) + letter    # S-1A
        audio_filename = "%s/%s/%s_R_16k.flac" % (ORIGINAL_DATA_PATH, subj_id, subj_id)
        split_filename = "%s/%s/%s_pedal_hand_corr.TextGrid" % (ORIGINAL_DATA_PATH, subj_id, subj_id)
        transcript_filename = "%s/%s/%s_R_16k-wrd.ctm" % (ORIGINAL_DATA_PATH, subj_id, subj_id)
        data, SAMPLE_RATE = sf.read(audio_filename)
        textgrid_obj = tg.TextGrid.load(split_filename)
        with open(transcript_filename, 'r') as transcript:
            line = transcript.readline().strip().split()
            for _, tier in enumerate(textgrid_obj):
                textgrid_lines = str(tier).split('\n')[1:]
                for i in xrange(len(textgrid_lines)):
                    clip_info = textgrid_lines[i].strip().split()
                    # skip clips that are unlabelled
                    if len(clip_info) != 3 or clip_info[2] == "START": 
                        continue
                    start_time = float(clip_info[0])
                    end_time = float(clip_info[1])

                    # trim silence from start and end of clip
                    while float(line[2]) < start_time:
                        line = transcript.readline().strip().split()                        
                    start_time = float(line[2])
                    if start_time > end_time: 
                        continue
                    print start_time, end_time
                    while len(line) > 0 and float(line[2]) + float(line[3]) < end_time:
                        temp = float(line[2]) + float(line[3])
                        line = transcript.readline().strip().split()
                    end_time = temp
                    start  = int(start_time * SAMPLE_RATE)
                    end = int(end_time * SAMPLE_RATE)
                    truth_value = clip_info[2] == "TRUTH"
                    clip = np.array(data[start : end])

                    # save clip as wav
                    wav_filename = "%s/%d_%d_%r.wav" % (WAV_CLIPS_PATH, subject_number, i, truth_value)
                    print wav_filename
                    scikits.audiolab.wavwrite(clip, wav_filename, fs=SAMPLE_RATE, enc='pcm32')
                    i += 1

                    # extract mfcc features and write to .npy
                    """
                    clip_mfcc = psf.mfcc(clip, SAMPLE_RATE)
                    if clip_mfcc.shape[0] >= 100:       
                        np.save("%s/%d_%d_%r.npy" % (CLIPS_MFCCS_PATH, subject_number, i, truth_value), clip_mfcc)
                    """

#split_audio()

def get_rnn_inputs(timestep_length = 0.2, truncate_length = 20):
    """ 
    prosodic + mfcc features - each clip split into quarter seconds
    """
    #os.system("rm " + RNN_INPUTS_ARFF_FILE)
    for filename in os.listdir(WAV_CLIPS_PATH):
        subject, clip, truth = filename[:-4].split('_')
        clip_data, SAMPLE_RATE = sf.read(WAV_CLIPS_PATH + "/" + filename)

        # truncated long clips (about 7% of data), split into timesteps
        if len(clip_data) > SAMPLE_RATE * truncate_length:
            clip_data = clip_data[: SAMPLE_RATE * truncate_length]
            print "TRUNCATED"
        timestep_size = int(timestep_length * SAMPLE_RATE)
        num_whole_timesteps = (len(clip_data) - 1) / timestep_size
        splits = np.array(range(1, num_whole_timesteps + 1)) * timestep_size
        timesteps = np.array_split(clip_data, splits)
        #print len(timesteps), len(timesteps[-1])
        for i in xrange(len(timesteps)):
            # save each timestep as a separate wav
            timestep_filename = "%s_%s_%d_%s" % (subject, clip, i, truth)
            wav_filepath = "%s/%s.wav" % (RNN_INPUTS_PATH, timestep_filename)
            scikits.audiolab.wavwrite(timesteps[i], wav_filepath, fs=SAMPLE_RATE, enc='pcm32')
            # extract opensmile features of each timestep to RNN_INPUTS_ARFF_FILE
            command = ' '.join([OPENSMILE_EXEC, "-C", OPENSMILE_CONFIG, 
                                                "-N", timestep_filename, 
                                                "-I", wav_filepath, 
                                                "-O", RNN_INPUTS_ARFF_FILE, 
                                                ">", "~/cs224s/log", "2>&1"])
            os.system(command)
            os.system("rm " + wav_filepath)
        print filename

get_rnn_inputs()

def get_rnn_inputs_lexical():
    # prosodic + mfcc + lexical features
    pass