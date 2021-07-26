import tensorflow as tf
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = np.array(x_train/255.0,dtype="<f4"), np.array(x_test/255.0,dtype="<f4")

pixel_mean=np.mean(x_train,axis=0)
x_train=x_train-pixel_mean
x_test=x_test-pixel_mean

idx=np.argsort(y_train.flatten())
vdx=np.array([5000*i+j for i in range(10) for j in range(4500,5000)])
tdx=np.array([5000*i+j for i in range(10) for j in range(4500)])

x_val, y_val = x_train[idx[vdx]] , y_train[idx[vdx]]
x_train, y_train = x_train[idx[tdx]] , y_train[idx[tdx]]

idx=np.random.permutation(45000)
x_train, y_train = x_train[idx] , y_train[idx]

idx=np.random.permutation(5000)
x_val, y_val = x_val[idx] , y_val[idx]

# for k in range(10):
#     print(np.sum((y_train==k).astype(int)), np.sum((y_val==k).astype(int)))

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=0.125,
        height_shift_range=0.125,
        fill_mode='nearest',
        horizontal_flip=True,
        validation_split=0.0)
