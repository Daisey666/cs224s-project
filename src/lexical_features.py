import pdb
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

WORD_VEC_PATH = "/afs/ir/users/j/w/jwlouie/cs224s/hw4/word_vector"
SENTENCES_FILENAME = "/afs/ir/users/j/w/jwlouie/cs224s/final-project/cs224s-project/raw_data/sentences.txt"
OUTPUT_FILENAME = "/afs/ir/users/j/w/jwlouie/cs224s/final-project/cs224s-project/raw_data/qa_glove_features.txt"

def parse_out_string(filename, searchstr):
    result = []
    with open(filename, "r") as f:
        for line in f.readlines():
            line = line.strip()

            # We're going to read out the answer from the interviewee
            if len(line) > 0 and line[0:len(searchstr)] == searchstr:
                result.append(line.split(":")[-1].split())
    return result

def parse_glove(filename):
    result = {}
    with open(filename, "r") as f:
        for line in f:
            line = line.split()
            result[line[0]] = map(float, line[1:])
    return result

def calculate_sentence_val(glove, sentence):
    sentence_val = [0.0 for i in xrange(300)]
    for word in sentence:
        if word in glove:
            sentence_val = [glove[word][i] + sentence_val[i] for i in xrange(300)]
    return [val / float(len(sentence_val)) for val in sentence_val]

def featurize_words():
    glove = parse_glove(WORD_VEC_PATH)
    print "parsed glove"
    answers = parse_out_string(SENTENCES_FILENAME, "answer")
    print "parsed answers"
    questions = parse_out_string(SENTENCES_FILENAME, "question")
    print "parsed questions"

    
    with open(OUTPUT_FILENAME, "w+") as output:

        for answer, question in zip(answers, questions):
            # Average word vector for answer:
            answer_glove = calculate_sentence_val(glove, answer)
            
            # Average word vector for question:
            question_glove = calculate_sentence_val(glove, question)

            # Create a 600d vector for our features
            result = answer_glove + question_glove

            output.write(" ".join(map(str, result)) + "\n")
        

featurize_words()
