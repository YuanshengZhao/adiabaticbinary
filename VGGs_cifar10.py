import tensorflow as tf
import numpy as np
import layers
import cifar10
import trainer
import sys


mtp=8
class VGG_SMALL(tf.keras.Model):
    def __init__(self,binW,binA):
        super(VGG_SMALL, self).__init__()
        self.binW=binW
        self.binA=binA
        self.conv01 = tf.keras.layers.Conv2D(16*mtp,3,strides=1,padding='same')
        self.conv02 = layers.BinaryConv2D(16*mtp,3,1,False) if binW else tf.keras.layers.Conv2D(16*mtp,3,strides=1,padding='same')
        self.conv03 = layers.BinaryConv2D(32*mtp,3,1,False) if binW else tf.keras.layers.Conv2D(32*mtp,3,strides=1,padding='same')
        self.conv04 = layers.BinaryConv2D(32*mtp,3,1,False) if binW else tf.keras.layers.Conv2D(32*mtp,3,strides=1,padding='same')
        self.conv05 = layers.BinaryConv2D(64*mtp,3,1,False) if binW else tf.keras.layers.Conv2D(64*mtp,3,strides=1,padding='same')
        self.conv06 = layers.BinaryConv2D(64*mtp,3,1,False) if binW else tf.keras.layers.Conv2D(64*mtp,3,strides=1,padding='same')
        self.dense = tf.keras.layers.Dense(10,activation='softmax',kernel_regularizer=tf.keras.regularizers.l2(5e-4))
        self.mpool02 = tf.keras.layers.MaxPooling2D()
        self.mpool04 = tf.keras.layers.MaxPooling2D()
        self.mpool06 = tf.keras.layers.MaxPooling2D()
        self.batnor01=tf.keras.layers.BatchNormalization()
        self.batnor02=tf.keras.layers.BatchNormalization()
        self.batnor03=tf.keras.layers.BatchNormalization()
        self.batnor04=tf.keras.layers.BatchNormalization()
        self.batnor05=tf.keras.layers.BatchNormalization()
        self.batnor06=tf.keras.layers.BatchNormalization()
        self.actv01=layers.BinaryActivation(False) if binA else tf.keras.layers.Activation("relu")
        self.actv03=layers.BinaryActivation(False) if binA else tf.keras.layers.Activation("relu")
        self.actv05=layers.BinaryActivation(False) if binA else tf.keras.layers.Activation("relu")
        self.actv02=layers.BinaryActivation(False) if binA else tf.keras.layers.Activation("relu")
        self.actv04=layers.BinaryActivation(False) if binA else tf.keras.layers.Activation("relu")
        self.actv06=tf.keras.layers.Activation('relu')
        self.fltn=tf.keras.layers.Flatten()
        self.drop01=tf.keras.layers.Dropout(.2)
        self.drop02=tf.keras.layers.Dropout(.2)
        self.drop03=tf.keras.layers.Dropout(.2)
        self.drop04=tf.keras.layers.Dropout(.2)
        self.drop05=tf.keras.layers.Dropout(.2)
        self.drop06=tf.keras.layers.Dropout(.2)

    def set_kk(self, kka):
        if not self.binW:
            raise NotImplementedError
        self.conv03.kk.assign(kka)
        self.conv05.kk.assign(kka)
        self.conv02.kk.assign(kka)
        self.conv04.kk.assign(kka)
        self.conv06.kk.assign(kka)

    def set_ka(self,kka):
        if not self.binA:
            raise NotImplementedError
        self.actv01.kk.assign(kka)
        self.actv03.kk.assign(kka)
        self.actv05.kk.assign(kka)
        self.actv02.kk.assign(kka)
        self.actv04.kk.assign(kka)
    
    def get_kk(self):
        if not self.binW:
            raise NotImplementedError
        return self.conv03.kk
    
    def get_ka(self):
        if not self.binA:
            raise NotImplementedError
        return self.actv01.kk

    # with drop out, statistics in training and validation will be different; we always use batch mean and var.
    # we deal with mean and var after training.
    def call(self, inputs, training=False):
        x=self.drop01(self.actv01(self.batnor01(self.conv01(inputs),training=True)),training=training)
        x=self.mpool02(self.conv02(x))
        x=self.drop02(self.actv02(self.batnor02(x,training=True)),training=training)
        x=self.drop03(self.actv03(self.batnor03(self.conv03(x),training=True)),training=training)
        x=self.mpool04(self.conv04(x))
        x=self.drop04(self.actv04(self.batnor04(x,training=True)),training=training)
        x=self.drop05(self.actv05(self.batnor05(self.conv05(x),training=True)),training=training)
        x=self.mpool06(self.conv06(x))
        x=self.drop06(self.actv06(self.batnor06(x,training=True)),training=training)
        return self.dense(self.fltn(x))

class VGG_SMALL_EVAL(VGG_SMALL):
    def __init__(self,binW,binA):
        super().__init__(binW,binA)
    def call(self, inputs, training=False):
        x=self.actv01(self.batnor01(self.conv01(inputs),training=training))
        x=self.mpool02(self.conv02(x))
        x=self.actv02(self.batnor02(x,training=training))
        x=self.actv03(self.batnor03(self.conv03(x),training=training))
        x=self.mpool04(self.conv04(x))
        x=self.actv04(self.batnor04(x,training=training))
        x=self.actv05(self.batnor05(self.conv05(x),training=training))
        x=self.mpool06(self.conv06(x))
        x=self.actv06(self.batnor06(x,training=training))
        return self.dense(self.fltn(x))

mode=sys.argv[1]

trr=trainer.Trainer(1000,mode)
trr.prto=1.5
trr.maxpush=3

if(mode=="w"):
    bw=1
    ba=0
    trr.lr_power=0
    # init_lr=.02
    # optz=tf.keras.optimizers.SGD(learning_rate=init_lr,momentum=0.9,nesterov=True,clipvalue=.01)
    init_lr=.001
    optz=tf.keras.optimizers.Adam()
elif(mode=="a"):
    bw=0
    ba=1
    init_lr=.001
    optz=tf.keras.optimizers.Adam()
elif(mode=="b"):
    bw=1
    ba=1
    init_lr=.001
    optz=tf.keras.optimizers.Adam()
else:
    print("mode = w|a|b")
    quit()

print("VGG Small, mode "+mode)

model=VGG_SMALL(bw,ba)
model(cifar10.x_train[:2])
model.compile(optimizer=optz,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# if(ba):  # activation +-1 if bias=0 0/1 if bias=1
#     model.actv01.bias.assign(0.0)
#     model.actv02.bias.assign(0.0)
#     model.actv03.bias.assign(0.0)
#     model.actv04.bias.assign(0.0)
#     model.actv05.bias.assign(0.0)

model.evaluate(cifar10.x_val,cifar10.y_val,batch_size=trr.val_bs,verbose=2)

trr.sw_epc=0
#pass tbest=false makes training faster but may reduce accurace a little
trr.train(model,cifar10.datagen,cifar10.x_train,cifar10.y_train,cifar10.x_val,cifar10.y_val,100,1e-3,"DVGG_",True,cifar10.x_test,cifar10.y_test)

trr.sw_epc=0
trr.same_wts_ep=30
#pass tbest=false makes training faster but may reduce accurace a little
trr.train(model,cifar10.datagen,cifar10.x_train,cifar10.y_train,cifar10.x_val,cifar10.y_val,600,1e-4,"DVGG_",True,cifar10.x_test,cifar10.y_test, mode != "b")

modelE=VGG_SMALL_EVAL(bw,ba)
modelE(cifar10.x_train[:2])
modelE.compile(optimizer=optz,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

with np.load("DVGG_"+ mode + "_Best.npz") as wts:
    modelE.set_weights([wts[nms] for nms in wts.files])

if(bw):
    modelE.set_kk(1e5)
if(ba):
    modelE.set_ka(1e5)

modelE.evaluate(cifar10.x_val,cifar10.y_val,batch_size=trr.val_bs,verbose=2)
x_trains=cifar10.x_train[:]
for _ in range(5):
    x_trains=x_trains[np.random.permutation(len(x_trains))]
    for ik in range(0,len(x_trains),trr.val_bs):
        modelE(x_trains[ik:ik+trr.val_bs],training=True)
        if (ik % 2000 == 0):
            print("#", end="")
    modelE.evaluate(cifar10.x_val,cifar10.y_val,batch_size=trr.val_bs,verbose=2)
    modelE.evaluate(cifar10.x_test,cifar10.y_test,batch_size=trr.val_bs,verbose=2)


if(mode != "a"):
    quit()
tf.keras.backend.clear_session()
bw=ba=1
init_lr=.001
optz=tf.keras.optimizers.Adam()
model=VGG_SMALL(bw,ba)
model(cifar10.x_train[:2])
model.compile(optimizer=optz,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

with np.load("DVGG_"+ mode + "_Best.npz") as wts:
    wtt=[wts[nms] for nms in wts.files]
for k in [1,3,5,7,9,11]:
    wtt[k]*=0
for k in [11,9,7,5,3]:
    wtt.insert(k,np.array(1.0))
for k in [17,14,11,8,5]:
    wtt.insert(k,np.array(1.0))

model.set_weights(wtt)

model.set_kk(1)
model.set_ka(1)

model.evaluate(cifar10.x_val,cifar10.y_val,batch_size=trr.val_bs,verbose=2)

mode="b"
trr.refresh()
trr.same_wts_ep=30
trr.mode="b"
#pass tbest=false makes training faster but may reduce accurace a little
trr.train(model,cifar10.datagen,cifar10.x_train,cifar10.y_train,cifar10.x_val,cifar10.y_val,600,init_lr*.1,"DVGG_",True,cifar10.x_test,cifar10.y_test,False)    

modelE=VGG_SMALL_EVAL(bw,ba)
modelE(cifar10.x_train[:2])
modelE.compile(optimizer=optz,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

with np.load("DVGG_"+ mode + "_Best.npz") as wts:
    modelE.set_weights([wts[nms] for nms in wts.files])

if(bw):
    modelE.set_kk(1e5)
if(ba):
    modelE.set_ka(1e5)

modelE.evaluate(cifar10.x_val,cifar10.y_val,batch_size=trr.val_bs,verbose=2)
x_trains=cifar10.x_train[:]
for _ in range(5):
    x_trains=x_trains[np.random.permutation(len(x_trains))]
    for ik in range(0,len(x_trains),trr.val_bs):
        modelE(x_trains[ik:ik+trr.val_bs],training=True)
        if (ik % 2000 == 0):
            print("#", end="")
    modelE.evaluate(cifar10.x_val,cifar10.y_val,batch_size=trr.val_bs,verbose=2)
    modelE.evaluate(cifar10.x_test,cifar10.y_test,batch_size=trr.val_bs,verbose=2)
