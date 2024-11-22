import tensorflow as tf
from tensorflow import keras
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


housing = fetch_california_housing()
all_x_train, x_test, all_y_train, y_test = train_test_split(housing.data, housing.target)
 
 
x_train, x_validation, y_train, y_validation = train_test_split(all_x_train, all_y_train)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_validation_scaled = scaler.transform(x_validation)
x_test_scaled = scaler.transform(x_test)


input = keras.layers.Input(shape=x_train_scaled.shape[1:])
hidden1 = keras.layers.Dense(30, activation="relu")(input)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
hidden3 = keras.layers.Dense(30, activation="relu")(hidden2)
output = keras.layers.Dense(1)(hidden3)

model = keras.models.Model(inputs=[input], outputs=[output])

model.compile(loss="mse", optimizer="rmsprop", metrics=["mae"])
ressults = model.fit(x_train_scaled, y_train, epochs=40, validation_data=(x_validation_scaled, y_validation))

res_eval = model.evaluate(x_test_scaled, y_test)

 

