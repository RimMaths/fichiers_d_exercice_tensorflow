import tensorflow as tf
from tensorflow import keras
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''Développement de réseaux de neurones ;
Premierement developpement de reseaux de neurones pour reconnaissances des formes à partir des  images 
Deuxiemement developper un modèle de régression avec un reseau de neurone '''

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#Jeu de données qui porte le nom de fashion_mnist
fashion_mnist_data = keras.datasets.fashion_mnist
(all_x_train, all_y_train), (x_test, y_test) = fashion_mnist_data.load_data()

#transormer pixel en float32
all_x_train = all_x_train.astype('float32') 
x_test = x_test.astype('float32')
 
# nombre d'information sur les images et chacune des images est représentée par une matrice  de 28*28
'''
print(f"all_x_train.shape = {all_x_train.shape}")
print(f"all_x_train[0].shape = {all_x_train[0].shape}")
print(f"all_x_train[0].dtype = {all_x_train[0].dtype}")
'''
 
# On créé un jeu de donné de validation et d'autre d'entrainement 
x_validation, x_train = all_x_train[:5000] / 255.0, all_x_train[5000:] / 255.0
y_validation, y_train = all_y_train[:5000], all_y_train[5000:]

print(f"x_train.shape = {x_train.shape}")
print(f"x_train[0].shape = {x_train[0].shape}")
print(f"x_train[0].dtype = {x_train[0].dtype}")

 

#On crée un tableau dans lequel on récupère  les informations de l'image

fashion_mnist_class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

 
for cls in range(10):
    print(cls, ":",fashion_mnist_class_names[y_train[cls]]) 
 
#Visualiser les 5 premières images  de l'ensemble d'entrainement

for i in range(5):
    my_img= x_train[i]
    my_img_class = y_train[i]
    my_img_class_name = fashion_mnist_class_names[my_img_class] 
    plt.imshow(my_img)
    plt.title(my_img_class_name)
    plt.show()   
