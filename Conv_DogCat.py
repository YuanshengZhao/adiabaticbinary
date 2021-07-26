import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(16)
tf.config.threading.set_intra_op_parallelism_threads(16)

import numpy as np
import layers
import cat_dog
import trainer
import sys

class CONVNET(tf.keras.Model):
    def __init__(self,binW,binA):
        super(CONVNET, self).__init__()
        self.binW=binW
        self.binA=binA
        self.conv1=layers.BinaryConv2D(16,3,1,False) if (binW and not binA) else tf.keras.layers.Conv2D(16,3, padding='same')
        self.conv2=layers.BinaryConv2D(32,3,1,False) if binW else tf.keras.layers.Conv2D(32,3, padding='same')
        self.conv3=layers.BinaryConv2D(64,3,1,False) if binW else tf.keras.layers.Conv2D(64,3, padding='same')
        self.dense=layers.BinaryDense(2) if (binW and not binA) else tf.keras.layers.Dense(2)
        self.actv1=layers.BinaryActivation() if binA else tf.keras.layers.Activation('relu')
        self.actv2=layers.BinaryActivation() if binA else tf.keras.layers.Activation('relu')
        self.actv3=layers.BinaryActivationRL() if (binA and not binW) else tf.keras.layers.Activation('relu')
        self.actv4=tf.keras.layers.Activation('softmax')
        self.pool1=tf.keras.layers.MaxPooling2D()
        self.pool2=tf.keras.layers.MaxPooling2D()
        self.pool3=tf.keras.layers.MaxPooling2D()
        self.pool4=tf.keras.layers.Flatten()
        # self.pool4=tf.keras.layers.GlobalAveragePooling2D()
        # self.drop1=tf.keras.layers.Dropout(.0)
        # self.drop2=tf.keras.layers.Dropout(.0)
        # self.drop3=tf.keras.layers.Dropout(.0)
        self.batn1=tf.keras.layers.BatchNormalization()
        self.batn2=tf.keras.layers.BatchNormalization()
        self.batn3=tf.keras.layers.BatchNormalization()

    def set_kk(self, kka):
        if not self.binW:
            raise NotImplementedError
        self.conv2.kk.assign(kka)
        self.conv3.kk.assign(kka)
        if not self.binA:
            self.conv1.kk.assign(kka)
            self.dense.kk.assign(kka)

    def set_ka(self,kka):
        if not self.binA:
            raise NotImplementedError
        self.actv1.kk.assign(kka)
        self.actv2.kk.assign(kka)
        if not self.binW:
            self.actv3.kk.assign(kka)
    
    def get_kk(self):
        if not self.binW:
            raise NotImplementedError
        return self.conv2.kk
    
    def get_ka(self):
        if not self.binA:
            raise NotImplementedError
        return self.actv1.kk


    def call(self, inputs, training=False):
        x=self.actv1(self.batn1(self.pool1(self.conv1(inputs)),training=True))
        x=self.actv2(self.batn2(self.pool2(self.conv2(x     )),training=True))
        x=self.actv3(self.batn3(self.pool3(self.conv3(x     )),training=True))
        return self.actv4(self.dense(self.pool4(x)))

class CONVNET_EVAL(CONVNET):
    def __init__(self,binW,binA):
        super().__init__(binW,binA)
    def call(self, inputs, training=False):
        x=self.actv1(self.batn1(self.pool1(self.conv1(inputs)),training=training))
        x=self.actv2(self.batn2(self.pool2(self.conv2(x     )),training=training))
        x=self.actv3(self.batn3(self.pool3(self.conv3(x     )),training=training))
        return self.actv4(self.dense(self.pool4(x)))


mode=sys.argv[1]

trr=trainer.Trainer(1000,mode)
trr.prto=1.5
trr.maxpush=3
trr.val_bs=100

if(mode=="w"): 
    bw=1
    ba=0
    initLR=.1
    optz=tf.keras.optimizers.SGD(learning_rate=initLR)
elif(mode=="a"):
    bw=0
    ba=1
    initLR=.01
    optz=tf.keras.optimizers.Adam(learning_rate=initLR)
    trr.lr_power=.3
elif(mode=="b"):
    bw=ba=1
    initLR=.01
    optz=tf.keras.optimizers.Adam(learning_rate=initLR)
    trr.lr_power=.3
else:
    print("mode = w|a|b")
    quit()

print("CONVNET, mode "+mode)

model=CONVNET(bw,ba)
model(cat_dog.x_train[:2])
model.compile(optimizer=optz,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

model.evaluate(cat_dog.x_val,cat_dog.y_val,batch_size=trr.val_bs,verbose=2)

trr.sw_epc=0
trr.train(model,cat_dog.datagen,cat_dog.x_train,cat_dog.y_train,cat_dog.x_val,cat_dog.y_val,80,initLR,"ConvDC_",True,cat_dog.x_test,cat_dog.y_test)

trr.sw_epc=0
trr.same_wts_ep=15
trr.train(model,cat_dog.datagen,cat_dog.x_train,cat_dog.y_train,cat_dog.x_val,cat_dog.y_val,600,initLR/10,"ConvDC_",True,cat_dog.x_test,cat_dog.y_test,mode!="b")

modelE=CONVNET_EVAL(bw,ba)
modelE(cat_dog.x_train[:2])
modelE.compile(optimizer=optz,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

with np.load("ConvDC_"+ mode + "_Best.npz") as wts:
    modelE.set_weights([wts[nms] for nms in wts.files])

if(bw):
    modelE.set_kk(1e5)
if(ba):
    modelE.set_ka(1e5)

modelE.evaluate(cat_dog.x_val,cat_dog.y_val,batch_size=trr.val_bs,verbose=2)
x_trains=cat_dog.x_train[:]
for _ in range(5):
    x_trains=tf.random.shuffle(x_trains)
    for ik in range(0,len(x_trains),trr.val_bs):
        modelE(x_trains[ik:ik+trr.val_bs],training=True)
        if (ik % 2000 == 0):
            print("#", end="")
    modelE.evaluate(cat_dog.x_val,cat_dog.y_val,batch_size=trr.val_bs,verbose=2)
    modelE.evaluate(cat_dog.x_test,cat_dog.y_test,batch_size=trr.val_bs,verbose=2)
