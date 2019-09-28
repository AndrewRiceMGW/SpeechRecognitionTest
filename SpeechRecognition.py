#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 22:33:05 2019

@author: rebecca
"""

import tflearn
import speech_data

lr = 0.001
epochs = 30000

batch = word_batch = speech_data.mfcc_batch_generator(64)
x, y = next(batch)
trainX, trainY = x, y
testX, testY, x, y
net = tflearn.input_data([None, 20, 80])
net = tflearn.lstm(net, 128, dropout = 0.8)
net = tflearn.fully_connected(net, 10, activation = 'softmax')
net = tflearn.regression(net, optimizer = 'adam', learning_rate = lr, 
                         loss = 'categorical_crossentropy')
model = tflearn.DNN(net, tensorboard_verbose=0)

while 1:
    model.fit(trainX, trainY, n_epoch=10, validation_set = (testX, testY), 
              show_metric =True, batch_size =64)
    _y = model.predict(x)
model.save('tflearn.lstm.model')

print(_y)
print(y)