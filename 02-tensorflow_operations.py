import tensorflow as tf
import os 

 

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


a=tf.constant(10.)
b = tf.constant(10.) 
#print(f"a+b = {a+b}")


c=tf.constant(10., dtype="float64")
#print(f"b+c = {b+c}") 

#print(f"b+c = {b+tf.cast(c, tf.float32 )}")



tensor_1 = tf.Variable([[1,2,3],[4,5,6],[7,8,9]])
'''
#print(f"tensor_1 = \n{tensor_1}")

tensor_1.assign(tensor_1 * 10)
#print(f"tensor_1 = \n{tensor_1}")

#la valeur  de la 2eme ligne et 2eme colonne sera affecté par 2021
tensor_1[1,1].assign(2021)
print(f"tensor_1 = \n{tensor_1}")
''' 
'''

tensor_2 = tf.constant([[1,2,3],[4,5,6],[7,8,9]])
tensor_2[1,1] = -10
print(tensor_2[1,1])


tensor_1[:,2].assign([-10,-10,-10])
print(f"tensor_1 = \n{tensor_1}")

'''


#Initialisation des tensors

'''
tensor_3 = tf.zeros((3,3))
print(f"tensor_3 = \n{tensor_3}")

'''
'''
tensor_4 = tf.random.uniform((3,3), minval=1, maxval=10, seed=2)
print(f"tensor_4 = \n{tensor_4}")

'''
#tensor_5 = tf.random.uniform((3,3), seed=2)
#print(f"tensor_5 = \n{tensor_5}")

#matrice identité 
#tensor_6 = tf.eye(3)
#print(f"tensor_6 = \n{tensor_6}")


#initialiser une matrice  avec des valeurs spécifiques

#tensor_7 = tf.fill((3,3), value = 5) 
#print(f"tensor_7 = \n{tensor_7}")


#Opérations algébriques

#Calcul sur les matrices ,transposé , multiplication 

#creation de tensor initialisé par random uniform 
tensor_8 = tf.random.uniform((3,3)) 
tensor_9 = tf.random.uniform((3,3)) 
print(f"tensor_8 = \n{tensor_8}")
print(f"tensor_9 = \n{tensor_9}")

#multiplication  matricielle

tensor_10 = tf.matmul(tensor_8, tensor_9)
print(f"tensor_10 = \n{tensor_10}")

#transposé  d'une matrice

tensor_11 = tf.transpose(tensor_8)
print(f"tensor_11 = \n{tensor_11}")





