'''Programme pour reconnaitre les objets de ces images de manière automatique 
pour voir l'utilité d'avoir 3 groupes de données test, traon , validation
le but : construire un reseau de neurone capable de reconnaitre la nature de ces vetements à partir de ces images 
l'architecture : MLP , multi layer perceptron , enchainement de couches les unes apres les autres en commençant par 
par couche d'entrée jusqu'à couche de sortie et en utilisant des couches cachées en options , connecté de manière linéaire'''

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

 

# Grace à la classe Sequential on a déclaré model , suceptible de réprenter un réseau de neurone avec Architecture MLP

model = keras.models.Sequential()
#Premiere couche  d'entrée , chacun des neurones va recevoir des données de dim 28*28
#en entréé matrice 28*28 , en sortie un tableau d'où Flatten à une dimension
model.add(keras.layers.Flatten(input_shape=[28, 28]))
#deuxieme  couche cachée , 300 neurones , fonction activation ReLU , le type de fonction d'activation
model.add(keras.layers.Dense(300, activation="relu"))
#3eme couche de  sortie , 150 neurones , fonction activation Relu
model.add(keras.layers.Dense(150, activation="relu"))
#couche de sortie
model.add(keras.layers.Dense(10, activation="softmax"))

#Afficher les info relatives à ce reseau de neurones
model.summary() 
