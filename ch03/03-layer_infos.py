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
#3eme couche , 2 eme couche cachée , 150 neurones , fonction activation Relu
#model.add(keras.layers.Dense(150, activation="relu"))
# declarer un objet d'initialisation des poids , mean :valeur moyenne de 100
my_weights_initializer = tf.keras.initializers.RandomNormal(mean=100., stddev=1.)
#Initialiser les biais par 1 , tensor dont toutes les valeurs sont egales à 1
my_bias_initializer= tf.keras.initializers.Ones()  
#Ajouter une couche avec 150 neurones  , fonction activation Relu , initialisation des poids et des biais*
model.add(keras.layers.Dense(150, activation="relu" ,\
                                  kernel_initializer = my_weights_initializer, \
                                  bias_initializer = my_bias_initializer))

#model.add(keras.layers.Dense(150, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

model.summary()
 
 
#print(model.layers)
#print(model.layers[2].name)

 
#On a récoupéré la deuxieme couche cachée
hidden_2 = model.get_layer('dense_1')
#Accès aux poids 
weights, biases = hidden_2.get_weights()
print("Les poids :\n ")
print(weights) 
print(weights.shape)
print("Les biais :\n ")
print(biases)
print(biases.shape)
 


'''  
my_weights_initializer = tf.keras.initializers.RandomNormal(mean=100., stddev=1.)
my_bias_initializer= tf.keras.initializers.Ones()  
model.add(keras.layers.Dense(150, activation="relu" ,\
                                  kernel_initializer = my_weights_initializer, \
                                  bias_initializer = my_bias_initializer))'''




#https://keras.io/initializers/.