import tensorflow as tf
from tensorflow import keras
import os
import pandas as pd
import numpy as np

 
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV 


os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

cifar10_class_names = {
    0:"Plane",
    1:"Car",
    2:"Bird",
    3:"Cat",
    4:"Deer",
    5:"Dog",
    6:"Frog",
    7:"Horse",
    8:"Boat",
    9:"Truck",
}




cifar10 = keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

 

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

x_train= x_train/255
x_test = x_test /255

 

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

 
 


def get_model(nb_hidden, nb_neurons):
    print(f"##  {nb_hidden}-{nb_neurons}   ##")
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[32,32,3])) 
    model.add(keras.layers.Dense(2 * nb_neurons, activation="relu"))    
    if nb_hidden > 1 :
        for layer in range(2, nb_hidden):
            model.add(keras.layers.Dense(nb_neurons, activation="relu"))
    model.add(keras.layers.Dense(10, activation="softmax" ))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"] )

    #model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"] )
    #model.compile(loss="mse", optimizer="rmsprop", metrics=["mae"])
    
    #https://keras.io/activations/
    #https://keras.io/losses/
    #https://keras.io/optimizers/ 
    #https://keras.io/metrics/

    return model


keras_classifier = keras.wrappers.scikit_learn.KerasClassifier(get_model)


param_distribs = {
"nb_hidden": [1,2,3,6,8],
"nb_neurons": [50,150,200],
}



early_stopping = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

search_cv = GridSearchCV(keras_classifier, param_distribs,  cv=2 )
search_cv.fit(x_train, y_train, epochs=200,  callbacks=[early_stopping], validation_data=[x_test, y_test],  )

print(f"search_cv.best_params_ = {search_cv.best_params_}")

print(f"search_cv.best_score_ = {search_cv.best_score_}") 

model = search_cv.best_estimator_.model
model.summary()

print(f"Evaluation : \n{model.evaluate(x_test, y_test)}")