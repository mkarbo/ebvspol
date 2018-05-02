#!/usr/bin/env python
# -*- coding: utf-8 -*-

#from __future__ import print_function

__author__ = 'mkarbo'

import numpy as np
import gensim
import string
import time

from keras.callbacks import LambdaCallback
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.utils.data_utils import get_file
import csv

def namefun():
    return input('input name ')

def csv2file(name):
    """
    Args:
        takes path to csv file

    Out:
        returns content of csv file as a list
    """
    with open(name, 'r') as csvfile:
        csvcontent = []
        content = csv.reader(csvfile, delimiter = ',')
        for row in content:
            csvcontent.append(row)
    return csvcontent

def parser(strlist):
    """
    Args:
        takes a list of strings

    Out:
        returns one strong of every string in the list appended with a
        linebreak
    """
    tempstr = strlist[0][0].strip()
    for i in range(1,len(strlist)):
        tempstr = tempstr + strlist[i][0] + '\r\n'
    return tempstr

def make_sentence(instr, n):
    """
    Args:
        takes a list of strings and an integer
    Out:
        returns a list of sentences of length n (#words) without punctuation
    """
    sentences = [\
                [word for word in\
                 doc[0].lower().translate(None, string.punctuation).split()[:n]]\
                for doc in instr]
    return sentences

def word2idx(word,word_model):
    """
    Arg:
        a word and a word_model ( Read the gensim docs on Word2Vec.)
    Out:
        word index
    """
    return word_model.wv.vocab[word].index

def idx2word(idx,word_model):
    """
    inverse of above
    """
    return word_model.wv.index2word[idx]

def sample(preds, temperature = 1.0):
    """
    classification function
    Arg:
        predictions, temperature
    Out:
        choice of predictor
    """
    if temperature <= 0:
        return np.argmax(preds)
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def avegsenlen(data):
    """
    Args:
        list of strings
    Out:
        average string length (#words)
    """
    data = [x[0].split() for x in data]
    return sum([len(x) for x in data]) / float(len(data))

if __name__ == '__main__':
    name = namefun()
    csvcontent = csv2file(name)
    print 'average len of sentence is' + str(avegsenlen(csvcontent))
    print 'number of sentences is ' + str(len(csvcontent))
    #csvcontent = parser(csvcontent)
    num_sen = input('number of sentences to use :' )
    sen_len = input('sentence length :')
    sentences = make_sentence(csvcontent, sen_len)[:num_sen]
    sentences = [sentence for sentence in sentences if len(sentence) > 0]
    word_model = gensim.models.Word2Vec(sentences,\
                                        size = 100,\
                                        min_count = 1,\
                                        window = 5,\
                                        iter = 100)
    pretrained_weights = word_model.wv.syn0
    wordlist = []
    for j in np.random.randint(0, len(word_model.wv.vocab.items()), 4):
        wordlist.append(word_model.wv.vocab.items()[j][0])
    vocab_size, embedding_size = pretrained_weights.shape
    for word in wordlist:
        most_similar = ', '.join('%s (%.2f)' % (similar, dist) for similar,
                                 dist in word_model.most_similar(word)[:8])
        print (' %s -> %s' % (word, most_similar))
    train_x = np.zeros([len(sentences), sen_len], dtype = np.int32)
    train_y = np.zeros([len(sentences)], dtype = np.int32)
    for i, sentence in enumerate(sentences):
        for t, word in enumerate(sentence[:-1]):
            train_x[i, t] = word2idx(word, word_model)
        train_y[i] = word2idx(sentence[-1], word_model)

    print '\r\n training Long Short Term Memory model'
    model = Sequential()
    model.add(Embedding(input_dim = vocab_size,\
                        output_dim = embedding_size,\
                        weights = [pretrained_weights]))
    model.add(LSTM(units = embedding_size))
    model.add(Dense(units = vocab_size))
    model.add(Activation('softmax'))
    model.compile(optimizer = 'adam',\
                  loss = 'sparse_categorical_crossentropy')

    def generate_next(text, num_generated = 10):
    """
    Args:
        takes in a string (one letter word) and optional an integer
    Out:
        a string of guesses from the model of length num_generated
    """
        word_idxs = [word2idx(word, word_model) for word in text.lower().split()]
        for i in range(num_generated):
            prediction = model.predict(x = np.array(word_idxs))
            idx = sample(prediction[-1], temperature = 0.7)
            word_idxs.append(idx)
        return ' '.join(idx2word(idx, word_model) for idx in word_idxs)
    answer = 'no'
    while answer == 'no':
        wordlist2 = []
        for j in np.random.randint(0,len(word_model.wv.vocab.items()),4):
            wordlist2.append(word_model.wv.vocab.items()[j][0])
        print wordlist2
        answer = input('satisfied? ')
        answer = str(answer)
    def on_epoch_end(epoch, _,wordlist2 = wordlist2):
    """
    Args:

    Out:

    """
        print '\r\n Generating text after epoch : %d' % epoch
        texts = wordlist2
        for text in texts:
            sample = generate_next(text)
            print '%s.. -> %s' % (text, sample)


    print '\r\n Fitting model on training data'
    model.fit(train_x, train_y,
              batch_size = 128,
              epochs = 20,
              callbacks = [LambdaCallback(on_epoch_end = on_epoch_end)])
    print 'saving model'
    model_json = model.to_json()
    with open('model4.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights('model4.h5')
    print '\r\n complete! '
