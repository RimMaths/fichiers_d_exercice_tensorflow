import tensorflow as tf 
import os
import numpy as np 

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

'''
a=tf.constant(10)
print(f"a = {a}")

''' 
B = tf.constant([[1.,2.,3],[4,5,6],[7,8,9]])

#print(f"B = {B}") 

'''
print(f"B.shape = {B.shape}")
print(f"B.dtype = {B.dtype}")

'''
 
#print(f"B[:,:] = {B[:,:]}") 

#print(f"B[1:,:] = {B[1:,:]}")

'''
print(f"B+100 = {B+100}")
print(f"B = {B}") 
'''


'''
print(f"B+100 = {tf.add(B,100)}")  
print(f"B = {B}")
'''

np_array_1 = np.array([1.,2.,3.,4.,5.,6.,7.,8.,9.,10.])
print(np_array_1)

 
tf_tensor = tf.constant(np_array_1)
print(tf_tensor) 

 
np_array_2 = tf_tensor.numpy()
print(np_array_2)



 