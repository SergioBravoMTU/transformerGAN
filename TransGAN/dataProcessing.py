import re
import json
import token
import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
import random
from sklearn.model_selection import train_test_split

def preprocess_sentence(sentence):
    # Change to lower case
    sentence = sentence.lower().strip()

    # delete xx-number-xx
    # sentence = "".join([word for word in sentence.split() if not word == "xx-number-xx"])

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."

    
    sentence = re.sub(r"xx-number-xx", r"#", sentence)

    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    # removing contractions
    sentence = re.sub(r"i'm", "i am", sentence)
    sentence = re.sub(r"he's", "he is", sentence)
    sentence = re.sub(r"she's", "she is", sentence)
    sentence = re.sub(r"it's", "it is", sentence)
    sentence = re.sub(r"that's", "that is", sentence)
    sentence = re.sub(r"what's", "that is", sentence)
    sentence = re.sub(r"where's", "where is", sentence)
    sentence = re.sub(r"how's", "how is", sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"won't", "will not", sentence)
    sentence = re.sub(r"can't", "cannot", sentence)
    sentence = re.sub(r"n't", " not", sentence)
    sentence = re.sub(r"n'", "ng", sentence)
    sentence = re.sub(r"'bout", "about", sentence)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
    sentence = sentence.strip()
    
    return sentence

def getStartAndEndTokens(X, y):
	
    # Build tokenizer using tfds for both questions and answers
    tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        X + y, target_vocab_size=2**13)

    # Define start and end token to indicate the start and end of a sentence
    START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

    return tokenizer, START_TOKEN, END_TOKEN

# Tokenize, filter and pad sentences
def tokenize_and_filter(inputs, outputs, tokenizer, START_TOKEN, END_TOKEN, MAX_LENGTH):
    tokenized_inputs, tokenized_outputs = [], []

    for (sentence1, sentence2) in zip(inputs, outputs):
        # tokenize sentence
        sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
        sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN
        # check tokenized sentence max length

        if len(sentence1) <= MAX_LENGTH and len(sentence2) <= MAX_LENGTH:
            tokenized_inputs.append(sentence1)
            tokenized_outputs.append(sentence2)

    # pad tokenized sentences
    tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
    tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_outputs, maxlen=MAX_LENGTH, padding='post')

    return tokenized_inputs, tokenized_outputs

def getDatasetTensor(X, y, BUFFER_SIZE, BATCH_SIZE):
    # decoder inputs use the previous target as input
    # remove START_TOKEN from targets
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'inputs': X,
            'dec_inputs': y[:, :-1]
        },
        {
            'outputs': y[:, 1:]
        },
    ))

    dataset = dataset.cache()
    dataset = dataset.shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

def getDatasetTensor_v2(X, dec_inputs, y, BUFFER_SIZE, BATCH_SIZE):
    # decoder inputs use the previous target as input
    # remove START_TOKEN from targets
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'input_1': X,
            'input_2': dec_inputs
        },
        {
            'outputs': y
        },
    ))

    dataset = dataset.cache()
    dataset = dataset.shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

def readDatafile(filepath):
    
    posts = []
    replies = []
    maxlength = 0

    with open(filepath, 'r') as f:
        for line in f.readlines():
            jsonline = json.loads(line)
            posts.append(preprocess_sentence(jsonline['post']))
            replies.append(preprocess_sentence(jsonline['reply']))

            maxlength = maxlength if maxlength >= len(preprocess_sentence(jsonline['reply'])) else len(preprocess_sentence(jsonline['reply']))

    print(maxlength)
    return posts, replies

def divideDataset(X, y, genTrain_size, val_size):

    X_gen, X_gan, y_gen, y_gan = train_test_split(X, y, 
                                                    train_size=genTrain_size, random_state=15)

    # X_gen_train, X_gen_val, y_gen_train, y_gen_val = train_test_split(X_gen, y_gen, 
    #                                                 train_size=val_size)

    # X_gan_train, X_gan_val, y_gan_train, y_gan_val = train_test_split(X_gan, y_gan, 
    #                                                 train_size=val_size)

    return X_gen, X_gan, y_gen, y_gan 

def onehotencode(X, tokenizer):
    def getOneHotVector(word):

        zeros = np.zeros(shape=(tokenizer.vocab_size+2))
        # print(zeros.shape)
        zeros[word]=1
        return zeros
    
    XOneHot = []
    for sentence in X:
        sentenceOneHot = np.array([getOneHotVector(word) for word in sentence])
        XOneHot.append(sentenceOneHot)
    # X = np.array([getOneHotVector(word) for sentence in y for word in sentence])
    # y = np.array([tokenizer.decode(word) for sentence in y for word in sentence])
    return np.array(XOneHot)