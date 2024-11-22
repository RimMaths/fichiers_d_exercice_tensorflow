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
 
#3 groupes de données de validation et de test 
x_train, x_validation, y_train, y_validation = train_test_split(all_x_train, all_y_train)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_validation_scaled = scaler.transform(x_validation)
x_test_scaled = scaler.transform(x_test)

#Constructio de réseau de neurone 
#3 couches d'entrées 
#constutiée de toutes les varibles du jeu de données
input_1 = keras.layers.Input(shape=x_train_scaled.shape[1:])
input_2 = keras.layers.Input(shape=[5])
input_3 = keras.layers.Input(shape=[3])

#couches cachées 
#1ere prend les entrées de la 1 ere couche d'entrée
hidden1 = keras.layers.Dense(30, activation="relu")(input_1)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
#objet intermidiare concaténation entre input2 et hidden2
concat_1 = keras.layers.concatenate([input_2, hidden2])

hidden3 = keras.layers.Dense(30, activation="relu")(concat_1)
concat_2 = keras.layers.concatenate([input_3, hidden3])

output = keras.layers.Dense(1)(concat_2)

model = keras.models.Model(inputs=[input_1, input_2, input_3], outputs=[output])

model.compile(loss="mse", optimizer="rmsprop", metrics=["mae"])
model.summary()

 

 

 
 
 
 
 

