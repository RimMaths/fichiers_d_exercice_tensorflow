import tensorflow as tf
from tensorflow import keras
import os
import pandas as pd
import numpy as np


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



model = keras.models.load_model("Models/cifar10_model.h5")


my_images = np.zeros((2, 32, 32, 3))


img1= keras.preprocessing.image.load_img("Images/dog.jpg", target_size=(32,32))
img2= keras.preprocessing.image.load_img("Images/cat.jpg", target_size=(32,32))
img1 = keras.preprocessing.image.img_to_array(img1) / 255
img2 = keras.preprocessing.image.img_to_array(img2) / 255


my_images[0, :] = img1
my_images[1, :] = img2


 
results = model.predict(my_images)
 
 
for r in results:
    predected_class_index = int(np.argmax(r)) 
    predected_prob = r[predected_class_index]
    print(f"{cifar10_class_names[predected_class_index]} - prob = {round(float(predected_prob),2)} ")
 

