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
from word2vec_lstm import .

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

    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    from keras.models import model_from_json

    model = model_from_json(loaded_model_json)
    model.load_weights('model.h5')
    num_new = input('number of new words to generate from')
    wordlist2 = []
    for j in np.random.randint(0,len(word_model.wv.vocab.items()),num_new):
        wordlist2.append(word_model.wv.vocab.items()[j][0])
    texts = wordlist2[:]
    for text in texts:
        predtext = generate_next(text)
        print text + '-> ' + predtext



