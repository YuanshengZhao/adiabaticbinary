import tensorflow as tf
import numpy as np
import layers
import mnist
import trainer
import sys


class DenseNet(tf.keras.Model):
    def __init__(self,binW,binA):
        super(DenseNet, self).__init__()
        self.binW=binW
        self.binA=binA
        self.flat= tf.keras.layers.Flatten()
        self.dense1 = layers.BinaryDense(128) if binW else tf.keras.layers.Dense(128)
        self.dense2 = layers.BinaryDense(10) if binW else tf.keras.layers.Dense(10)
        self.actv1=layers.BinaryActivationRL() if binA else tf.keras.layers.Activation("relu")
        self.actv2=tf.keras.layers.Activation("softmax")
        self.drop=tf.keras.layers.Dropout(.2)

    def set_kk(self, kka):
        if not self.binW:
            raise NotImplementedError
        self.dense1.set_kk(kka)
        self.dense2.set_kk(kka)

    def set_ka(self,kka):
        if not self.binA:
            raise NotImplementedError
        self.actv1.set_kk(kka)
    
    def get_kk(self):
        if not self.binW:
            raise NotImplementedError
        return self.dense1.kk
    
    def get_ka(self):
        if not self.binA:
            raise NotImplementedError
        return self.actv1.kk

    def call(self, inputs, training=False):
        x=self.drop(self.actv1(self.dense1(self.flat(inputs))),training=training)
        return self.actv2(self.dense2(x))

bw=int(sys.argv[1])
ba=int(sys.argv[2])
if(bw and not ba):
    mode="w"
elif(ba and not bw):
    mode="a"
else:
    mode="b"

print("VGG Small, mode "+mode)

model=DenseNet(bw,ba)
optz=tf.keras.optimizers.Adam()
model(mnist.x_train[:2])
model.compile(optimizer=optz,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

trr=trainer.Trainer(1000,mode)
trr.val_bs=32
trr.lr_power=.2

model.evaluate(mnist.x_val,mnist.y_val,batch_size=trr.val_bs,verbose=2)

if(mode=="a"):
    trr.sw_epc=0
    trr.lr_base=1.5
    trr.lr_power=.3
    model.set_ka(trr.lr_base)
    trr.train(model,None,mnist.x_train,mnist.y_train,mnist.x_val,mnist.y_val,8,1e-3,"Dse_",True,mnist.x_test,mnist.y_test)
    model.set_ka(6)
    trr.train(model,None,mnist.x_train,mnist.y_train,mnist.x_val,mnist.y_val,3,1e-3,"Dse_",True,mnist.x_test,mnist.y_test)
    model.set_ka(1001)
    trr.train(model,None,mnist.x_train,mnist.y_train,mnist.x_val,mnist.y_val,2,1e-3,"Dse_",True,mnist.x_test,mnist.y_test)
if(mode=="w"):
    trr.sw_epc=0
    trr.lr_base=3.0
    model.set_kk(trr.lr_base)
    trr.train(model,None,mnist.x_train,mnist.y_train,mnist.x_val,mnist.y_val,8,1e-3,"Dse_",True,mnist.x_test,mnist.y_test)
    for kkz in [10, 20, 50, 100, 300, 500, 999]:
        model.set_kk(kkz)
        trr.train(model,None,mnist.x_train,mnist.y_train,mnist.x_val,mnist.y_val,3,1e-3,"Dse_",True,mnist.x_test,mnist.y_test)
if(mode=="b"):
    trr.sw_epc=0
    trr.lr_base=3.0
    model.set_kk(trr.lr_base)
    model.set_ka(trr.lr_base)
    trr.train(model,None,mnist.x_train,mnist.y_train,mnist.x_val,mnist.y_val,8,1e-3,"Dse_",True,mnist.x_test,mnist.y_test)
    for kkz in [10, 20, 50, 100, 300, 500, 999]:
        model.set_ka(kkz)
        model.set_kk(kkz)
        trr.train(model,None,mnist.x_train,mnist.y_train,mnist.x_val,mnist.y_val,3,1e-3,"Dse_",True,mnist.x_test,mnist.y_test)
