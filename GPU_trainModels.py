#coding=utf-8
# show GPU info
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

#Import modules
import cv2
import pandas as pd
import numpy as np
import os
from numpy import asarray
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, hamming_loss, zero_one_loss
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input,Dense, Dropout, Activation, Flatten, Conv2D, BatchNormalization, Reshape, MaxPooling2D, CuDNNLSTM, Embedding, Concatenate

import keras
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks

#import data
import pickle
X = pickle.load(open("X_500.pickle", "rb")) #import preprocessed 500*500 data
Y = pickle.load(open("Y.pickle", "rb"))
X = X/255
width, height, channels = X.shape[1:]

Ycate = [np.argmax(i) for i in Y]
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced',np.unique(Ycate),Ycate)

data = pd.read_csv("data_processed.csv")

# prepare text data
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
np.random.RandomState(0)
t = Tokenizer()
X_text = data['corrected_text']
X_text = [str(i) for i in X_text]
t.fit_on_texts(X_text)
vocab_size = len(t.word_index) + 1

encoded_train = t.texts_to_sequences(X_text)
max_length = max(list(map(lambda x: len(x), encoded_train)))
padded_train = pad_sequences(encoded_train, maxlen=max_length, padding='post')
print("Image data shape:",X.shape)
print("Text data length:",padded_train.shape)
print("Y shape:",Y.shape)

#Build the text+img model
a = input_text = Input(shape = (max_length,))
a = Embedding(vocab_size, 100)(a)
a = CuDNNLSTM(128, return_sequences=True)(a)
a = CuDNNLSTM(128)(a)
a = Flatten()(a)
a = Dense(128, activation='relu')(a)

b = input_img = Input(shape = (width, height, channels))
b = Conv2D(32, (3, 3), activation='relu')(b)
b = MaxPooling2D((2,2))(b)
b = Conv2D(64, (3, 3), activation='relu')(b)
b = MaxPooling2D((2,2))(b)
b = Flatten()(b)
b = Dense(128, activation='relu')(b)

combine = Concatenate(axis=-1,)([a,b])
# combine = Dense(64, activation='relu')(combine)
# combine = Dropout(0.2)(combine)
# combine = Dense(32, activation='relu')(combine)
# combine = Dropout(0.2)(combine)
output = Dense(5, activation='softmax')(combine)
Model_text_img_clr = Model([input_text,input_img],output)
Model_text_img_clr.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
Model_text_img_clr.summary()

results = Model_text_img_clr.fit([padded_train,X], Y,
         epochs=20,
         batch_size=64,
         shuffle=True,
        validation_split=0.2,
        class_weight=class_weights
)

plt.plot(list(results.history.values())[0],'b',label='Train-Loss')
plt.plot(list(results.history.values())[1],'g',label='Train-Accuracy')
plt.legend(loc='upper center', shadow=True)

plt.plot(list(results.history.values())[2],'b',label='Test-Loss')
plt.plot(list(results.history.values())[3],'g',label='Test-Accuracy')
plt.legend(loc='upper center', shadow=True)

