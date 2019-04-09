#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 21:31:35 2019

@author: multiproxy
"""

import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist #28X28 images of handwritten digits between 0-9

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1) #to normalize (convert the values between 0-1) values
x_test = tf.keras.utils.normalize(x_test, axis=1) #to normalize (convert the values between 0-1) values

plt.imshow(x_train[0], cmap=plt.cm.binary)
model = tf.keras.models.Sequential() #defining the model
model.add(tf.keras.layers.Flatten()) # flatten reshapes the input to a vector / this is input layer
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) #first paramater here is the number of neurons and 
#the second one is the activation function / this is the first hidden layer
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) # /second hidden layer consists 128 neurons
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) # /third hidden layer consists 128 neurons
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) # /this is the output layer consists 10 neurons and
# the activation function used in this layer is softmax because this is a proboblity distibution

model.compile( optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=3)

