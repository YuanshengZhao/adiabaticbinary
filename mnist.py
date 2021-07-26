import tensorflow as tf
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = np.array(x_train/255.0,dtype="<f4"), np.array(x_test/255.0,dtype="<f4")

idx=np.argsort(y_train.flatten())
vdx=np.array([6000*i+j for i in range(10) for j in range(5400,6000)])
tdx=np.array([6000*i+j for i in range(10) for j in range(5400)])

x_val, y_val = x_train[idx[vdx]] , y_train[idx[vdx]]
x_train, y_train = x_train[idx[tdx]] , y_train[idx[tdx]]

idx=np.random.permutation(54000)
x_train, y_train = tf.convert_to_tensor(x_train[idx]) , tf.convert_to_tensor(y_train[idx])

idx=np.random.permutation(6000)
x_val, y_val = tf.convert_to_tensor(x_val[idx]) , tf.convert_to_tensor(y_val[idx])

x_test, y_test = tf.convert_to_tensor(x_test) , tf.convert_to_tensor(y_test)