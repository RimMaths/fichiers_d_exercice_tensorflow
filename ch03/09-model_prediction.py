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

 

model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
 

results = model.fit(x_train, y_train, epochs=2,\
                    validation_data=(x_validation, y_validation))

 

 
x_test= x_test/255.0
print("Evaluation du modèle :")
model.evaluate(x_test, y_test)


 

#Utiliser le modèle pour réaliser des prédictions : 

x_new = x_test[11:20]
y_prob = model.predict(x_new)
print(f"y_prob = {y_prob.round(2)}")
y_prediction = model.predict_classes(x_new)
print(f"y_prediction = {y_prediction}")

print(f"       Prédictions : {np.array(fashion_mnist_class_names)[y_prediction]}")
y_truth = y_test[11:20]
print(f"Les vraies classes : {np.array(fashion_mnist_class_names)[y_truth]}")