import tensorflow as tf

IMG_HEIGHT = IMG_WIDTH = 64

datagen = tf.keras.preprocessing.image.ImageDataGenerator(dtype=float,
                                                                    horizontal_flip=True,
                                                                    width_shift_range=.125,
                                                                    height_shift_range=.125,
                                                                    fill_mode="nearest",
                                                                    )
def loadData(path):
    ds = tf.keras.preprocessing.image_dataset_from_directory(path,
                                                            image_size=(IMG_HEIGHT, IMG_WIDTH),
                                                            batch_size=100,
                                                            color_mode= "grayscale",
                                                            interpolation='nearest'
                                                            )
    # load data to memory
    x_t=[]
    y_t=[]
    for image_batch, labels_batch in ds:
        x_t.append(image_batch)
        y_t.append(labels_batch)
    x_t=tf.concat(x_t,axis=0)
    y_t=tf.concat(y_t,axis=0)
    return tf.cast(x_t,tf.float32)/255., y_t

x_train,y_train = loadData("Ddogs-vs-cats/train")
x_val,y_val = loadData("Ddogs-vs-cats/validation")
x_test,y_test = loadData("Ddogs-vs-cats/test")

pixel_mean=tf.reduce_mean(x_train,axis=0)
x_train=x_train-pixel_mean
x_test=x_test-pixel_mean
x_val=x_val-pixel_mean

print(x_train.shape, x_val.shape, x_test.shape)
