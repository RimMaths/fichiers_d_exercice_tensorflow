#Reseau de neurone pour un modèle de régression 
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

#telecharger tout les données du jeu de donnée california housing
#Jeu d'entrainement
housing = fetch_california_housing()
all_x_train, x_test, all_y_train, y_test = train_test_split(housing.data, housing.target)

'''
print(f"all_x_train.shape = {all_x_train.shape}" )
print(f"all_x_train.dtype = {all_x_train.dtype}" )
print(f"type(all_x_train) = {type(all_x_train)}" )
print(f"Features = {all_x_train[0]} ; variable cible = {all_y_train[0]}" )
'''
''' L'affichage de ces 4 print :
all_x_train.shape = (15480, 8) ; dans ce jeu de données j'ai 15480 appartement renseignés avec 8 variables la 9 eme variable est dtockée dans all_y_train
all_x_train.dtype = float64 ; type associé aux données 
type(all_x_train) = <class 'numpy.ndarray'> ; tableu numpy 
Features = [ 2.32410000e+00  4.30000000e+01  3.55696203e+00  9.40928270e-01
  1.59000000e+03  3.35443038e+00  3.39600000e+01 -1.18210000e+02] ; variable cible = 1.593 faux multiplier par 100 000 pour avoir la vraie valeur'''

'''
#https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset
'''
#jeu de validation 
x_train, x_validation, y_train, y_validation = train_test_split(all_x_train, all_y_train)
scaler = StandardScaler()
#d'entrainement
x_train_scaled = scaler.fit_transform(x_train)
#de validation
x_validation_scaled = scaler.transform(x_validation)
#test
x_test_scaled = scaler.transform(x_test)
 

'''
print(f"Le MAX - MIN de y_train = [{np.max(y_train)}-{np.min(y_train)}]")
print(f"Le MAX - MIN de y_validation = [{np.max(y_validation)}-{np.min(y_train)}]")
print(f"Le MAX - MIN de y_test = [{np.max(y_test)}-{np.min(y_test)}]")
'''
'''Affichage : 
Y'a pas de difference entre le min et le max dans chaque jeu de donné ce qui est bien 
Le MAX - MIN de y_train = [5.00001-0.14999]
Le MAX - MIN de y_validation = [5.00001-0.14999]
Le MAX - MIN de y_test = [5.00001-0.14999]
'''
#Construction de réseau de neurones 

model = keras.models.Sequential() 
model.add(keras.layers.Dense(30, activation="relu", input_shape=x_train_scaled.shape[1:]))   
model.add(keras.layers.Dense(15, activation="relu"))
model.add(keras.layers.Dense(8, activation="relu"))
#Couche de sortie composée d'un unique neurone; besoin de prédire une seule valeur ; valeur quelconque pas besoin de fonction d'activation
model.add(keras.layers.Dense(1))



#model.summary()
 


model.compile(loss="mse", optimizer="rmsprop", metrics=["mae"])
ressults = model.fit(x_train_scaled, y_train, epochs=100, \
                     validation_data=(x_validation_scaled, y_validation))

mae_test = model.evaluate(x_test_scaled, y_test)



