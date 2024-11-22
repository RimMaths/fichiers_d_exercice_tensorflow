import tensorflow as tf
from tensorflow import keras
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


fashion_mnist_data = keras.datasets.fashion_mnist
(all_x_train, all_y_train), (x_test, y_test) = fashion_mnist_data.load_data()

all_x_train = all_x_train.astype('float32') 
x_test = x_test.astype('float32')
 
 


x_validation, x_train = all_x_train[:5000] / 255.0, all_x_train[5000:] / 255.0
y_validation, y_train = all_y_train[:5000], all_y_train[5000:]
 
 
 
fashion_mnist_class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

 

 
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(150, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

 
#Compilation

model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
 


 
#Training 

'''
results = model.fit(x_train, y_train, epochs=10,\
                    validation_data=(x_validation, y_validation))''' 

 
'''
#class_weight tableau 
results = model.fit(x_train, y_train, epochs=5,\
                    validation_data=(x_validation, y_validation),\
                    class_weight=[10,10,10,10,5,10,10,10,10,10]) 

 
'''
'''
results = model.fit(x_train, y_train, epochs=5,\
                    validation_data=(x_validation, y_validation),\
                    sample_weight=[....])
'''
