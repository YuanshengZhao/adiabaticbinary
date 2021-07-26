import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(16)
tf.config.threading.set_intra_op_parallelism_threads(16)

import numpy as np
import layers
import trainer
import sys

with np.load("../numbers.npz") as fp:
    x_train=tf.convert_to_tensor(fp["x_train"])
    y_train=tf.convert_to_tensor(fp["y_train"])
    x_val  =tf.convert_to_tensor(fp["x_val"])
    y_val  =tf.convert_to_tensor(fp["y_val"])
    x_test =tf.convert_to_tensor(fp["x_test"])
    y_test =tf.convert_to_tensor(fp["y_test"])

class CONVNET(tf.keras.Model):
    def __init__(self,binW,binA):
        super(CONVNET, self).__init__()
        self.binW=binW
        self.binA=binA
        self.conv1=layers.BinaryConv1D(16,30,1,False) if (binW and not binA) else tf.keras.layers.Conv1D(16,30, padding='same')
        self.conv2=layers.BinaryConv1D(32,30,1,False) if binW else tf.keras.layers.Conv1D(32,30, padding='same')
        self.conv3=layers.BinaryConv1D(64,30,1,False) if binW else tf.keras.layers.Conv1D(64,30, padding='same')
        self.dense=layers.BinaryDense(10) if (binW and not binA) else tf.keras.layers.Dense(10)
        self.actv1=layers.BinaryActivation() if binA else tf.keras.layers.Activation('relu')
        self.actv2=layers.BinaryActivation() if binA else tf.keras.layers.Activation('relu')
        self.actv3=layers.BinaryActivation() if (binA and not binW) else tf.keras.layers.Activation('relu')
        self.actv4=tf.keras.layers.Activation('softmax')
        self.pool1=tf.keras.layers.MaxPooling1D(10)
        self.pool2=tf.keras.layers.MaxPooling1D(8)
        self.pool3=tf.keras.layers.MaxPooling1D(5)
        self.pool4=tf.keras.layers.GlobalAveragePooling1D()
        self.drop1=tf.keras.layers.Dropout(.2)
        self.drop2=tf.keras.layers.Dropout(.2)
        self.drop3=tf.keras.layers.Dropout(.2)
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
        x=self.drop1(self.actv1(self.batn1(self.pool1(self.conv1(inputs)),training=True)),training=training)
        x=self.drop2(self.actv2(self.batn2(self.pool2(self.conv2(x     )),training=True)),training=training)
        x=self.drop3(self.actv3(self.batn3(self.pool3(self.conv3(x     )),training=True)),training=training)
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
model(x_train[:2])
model.compile(optimizer=optz,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

model.evaluate(x_val,y_val,batch_size=trr.val_bs,verbose=2)

trr.sw_epc=0
trr.train(model,None,x_train,y_train,x_val,y_val,80,initLR,"ConvDS_",True,x_test,y_test)

trr.sw_epc=0
trr.same_wts_ep=15
trr.train(model,None,x_train,y_train,x_val,y_val,600,initLR/10,"ConvDS_",True,x_test,y_test)

modelE=CONVNET_EVAL(bw,ba)
modelE(x_train[:2])
modelE.compile(optimizer=optz,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# with np.load("ConvDS__a7.6.npz") as wts:
#     modelE.set_weights([wts[nms] for nms in wts.files])
with np.load("ConvDS_"+ mode + "_Best.npz") as wts:
    modelE.set_weights([wts[nms] for nms in wts.files])

if(bw):
    modelE.set_kk(1e5)
if(ba):
    modelE.set_ka(1e5)

modelE.evaluate(x_val,y_val,batch_size=trr.val_bs,verbose=2)
x_trains=x_train[:]
for _ in range(5):
    x_trains=tf.random.shuffle(x_trains)
    for ik in range(0,len(x_trains),trr.val_bs):
        modelE(x_trains[ik:ik+trr.val_bs],training=True)
        if (ik % 2000 == 0):
            print("#", end="")
    modelE.evaluate(x_val,y_val,batch_size=trr.val_bs,verbose=2)
    modelE.evaluate(x_test,y_test,batch_size=trr.val_bs,verbose=2)
