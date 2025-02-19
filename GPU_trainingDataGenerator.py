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

# import tensorflow as tf
# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import Input,Dense, Dropout, Activation, Flatten, Conv2D, BatchNormalization, Reshape, MaxPooling2D, CuDNNLSTM, Embedding, Concatenate
#
# import keras
from keras.utils import to_categorical
# from keras.preprocessing.image import ImageDataGenerator
# from keras import callbacks

#Import data
PATH_CSV = "./data_7000_corrected.csv" # use pre-processed data, it ignores 26 missing images, actual enties are 6974
PATH_IMG = "./data_7000/"
data = pd.read_csv(PATH_CSV, header=None)
data.columns = ['Image_name', 'Image_URL', 'OCR_extracted_text', 'corrected_text','Humour', 'Sarcasm', 'offensive', 'Motivational', 'Overall_Sentiment']
print("Data shape:",data.shape)

## use cv2, and correct 2-channel error image
size = 500,500  # img size: height, width
c = 0
X = [ ]     # creating an empty array
error = []
img_width = []
img_height = []
for img_name in tqdm(data['Image_name']):
  c += 1
  if(os.path.isfile(PATH_IMG + img_name)):
    image = cv2.imread((PATH_IMG + img_name), cv2.IMREAD_COLOR)
    if image is not None:
      img_height.append(image.shape[0])
      img_width.append(image.shape[1])
      img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      img = cv2.resize(img, size)
      if img.ndim != 3:                  # if the image is not RGB
        img = np.dstack([img, img, img]) # use the same value for the 3 RGB channels
      X.append(img)  # storing each image in array X
    else:
      print('None:{}'.format(img_name))
      error.append(img_name)
  else:
    print('Not Exists:{}'.format(img_name))
    error.append(img_name)
print("Error Image:",len(error))

X = np.array(X)   # converting list to array

data = data[~data['Image_name'].isin(error)] # delete data of error images
data['img_width'] = img_width
data['img_height'] = img_height
Y = data['Overall_Sentiment']
num_class = 5
Y = to_categorical(Y.factorize()[0])

# export X, Y data
import pickle

pickle_out = open("X_{}.pickle".format(size[0]), "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("Y.pickle", "wb")
pickle.dump(Y, pickle_out)
pickle_out.close()

data.to_csv('data_processed.csv',index=False)


