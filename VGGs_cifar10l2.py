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
        self.conv02 = layers.BinaryConv2DCL(16*mtp,3,1,False) if binW else tf.keras.layers.Conv2D(16*mtp,3,strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(5e-4))
        self.conv03 = layers.BinaryConv2DCL(32*mtp,3,1,False) if binW else tf.keras.layers.Conv2D(32*mtp,3,strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(5e-4))
        self.conv04 = layers.BinaryConv2DCL(32*mtp,3,1,False) if binW else tf.keras.layers.Conv2D(32*mtp,3,strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(5e-4))
        self.conv05 = layers.BinaryConv2DCL(64*mtp,3,1,False) if binW else tf.keras.layers.Conv2D(64*mtp,3,strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(5e-4))
        self.conv06 = layers.BinaryConv2DCL(64*mtp,3,1,False) if binW else tf.keras.layers.Conv2D(64*mtp,3,strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.l2(5e-4))
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
trr.prto=1.2
trr.maxpush=3

if(mode=="w"):
    bw=1
    ba=0
elif(mode=="a"):
    bw=0
    ba=1
else:
    print("mode = w|a|b")
    quit()

print("VGG Small, mode "+mode)

if(bw==1 and ba==0):
    model=VGG_SMALL(0,0)
    optz=tf.keras.optimizers.SGD(learning_rate=.02,momentum=0.9,nesterov=True)
    model(cifar10.x_train[:2])
    model.compile(optimizer=optz,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    model.summary()
    model.evaluate(cifar10.x_val,cifar10.y_val,batch_size=125,verbose=2)
    model.fit(cifar10.datagen.flow(cifar10.x_train,cifar10.y_train,batch_size=100),epochs=80,validation_data=(cifar10.x_val,cifar10.y_val),verbose=2)
    np.savez("DVGG_l2fp1.npz",*(model.get_weights()))
    optz.learning_rate.assign(2e-3)
    model.fit(cifar10.datagen.flow(cifar10.x_train,cifar10.y_train,batch_size=100),epochs=40,validation_data=(cifar10.x_val,cifar10.y_val),verbose=2)
    np.savez("DVGG_l2fp2.npz",*(model.get_weights()))
    optz.learning_rate.assign(2e-4)
    model.fit(cifar10.datagen.flow(cifar10.x_train,cifar10.y_train,batch_size=100),epochs=20,validation_data=(cifar10.x_val,cifar10.y_val),verbose=2)
    np.savez("DVGG_l2fp3.npz",*(model.get_weights()))

    with np.load("DVGG_l2fp3.npz") as wts:
        wtt=[wts[nms] for nms in wts.files]

    for k in [11,9,7,5,3]: #nmk
        wtt.insert(k,np.array(tf.reduce_max(tf.math.abs(wtt[k-1])).numpy()*1.5))
    for k in [2,5,8,11,14]: # reset kernel
        wtt[k]=tf.math.atanh(wtt[k]/tf.reduce_max(tf.math.abs(wtt[k]))/1.5).numpy()
    for k in [1,4,7,10,13,16]: # remove bias
        wtt[k]*=0
    for k in [17,14,11,8,5]: # kk
        wtt.insert(k,np.array(1.0))

    tf.keras.backend.clear_session()
    model=VGG_SMALL(1,0)
    optz=tf.keras.optimizers.SGD(learning_rate=.002,momentum=0.9,nesterov=True,clipvalue=.01)
    model(cifar10.x_train[:2])
    model.compile(optimizer=optz,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    model.summary()
    model.set_weights(wtt)
    model.evaluate(cifar10.x_val,cifar10.y_val,batch_size=125,verbose=2)

    trr=trainer.Trainer(30,"w")
    trr.prto=1.5 # or 1.2
    trr.maxpush=3
    trr.target_acc=.94 # setting this to 0.93 makes training faster 
    trr.lr_power=0

    #pass tbest=false makes training faster but may reduce accurace a little
    trr.train(model,cifar10.datagen,cifar10.x_train,cifar10.y_train,cifar10.x_val,cifar10.y_val,600,2e-3,"DVGG_l2",True,cifar10.x_test,cifar10.y_test)

    with np.load("DVGG_l2" + mode + "_Best.npz") as wts:
        model.set_weights([wts[nms] for nms in wts.files])

    model.set_kk(1e5)
    model.evaluate(cifar10.x_val,cifar10.y_val,batch_size=trr.val_bs,verbose=2)
    model.evaluate(cifar10.x_test,cifar10.y_test,batch_size=trr.val_bs,verbose=2)

    x_trains=cifar10.x_train[:]
    for _ in range(5):
        x_trains=x_trains[np.random.permutation(len(x_trains))]
        for ik in range(0,len(x_trains),trr.val_bs):
            model(x_trains[ik:ik+trr.val_bs],training=True)
            if (ik % 2000 == 0):
                print("#", end="")
        model.evaluate(cifar10.x_val,cifar10.y_val,batch_size=trr.val_bs,verbose=2)
        model.evaluate(cifar10.x_test,cifar10.y_test,batch_size=trr.val_bs,verbose=2)

if(bw==0 and ba==1):
    model=VGG_SMALL(0,1)
    init_lr=.02
    optz=tf.keras.optimizers.SGD(learning_rate=init_lr,momentum=0.9,nesterov=True,clipvalue=.01)
    model(cifar10.x_train[:2])
    model.compile(optimizer=optz,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    model.summary()
    model.evaluate(cifar10.x_val,cifar10.y_val,batch_size=125,verbose=2)

    trr=trainer.Trainer(1000,"a")
    trr.prto=1.5
    trr.maxpush=3
    trr.target_acc=.94
    trr.lr_power=0.3
    trr.train(model,cifar10.datagen,cifar10.x_train,cifar10.y_train,cifar10.x_val,cifar10.y_val,100,init_lr    ,"DVGG_l2",True,cifar10.x_test,cifar10.y_test)
    trr.train(model,cifar10.datagen,cifar10.x_train,cifar10.y_train,cifar10.x_val,cifar10.y_val,40, init_lr*.1 ,"DVGG_l2",True,cifar10.x_test,cifar10.y_test)
    trr.train(model,cifar10.datagen,cifar10.x_train,cifar10.y_train,cifar10.x_val,cifar10.y_val,20, init_lr*.01,"DVGG_l2",True,cifar10.x_test,cifar10.y_test)

    trr.sw_epc=10
    trr.same_wts_ep=30
    trr.train(model,cifar10.datagen,cifar10.x_train,cifar10.y_train,cifar10.x_val,cifar10.y_val,600,init_lr*.1,"DVGG_l2",True,cifar10.x_test,cifar10.y_test)

    with np.load("DVGG_l2" + mode + "_Best.npz") as wts:
        model.set_weights([wts[nms] for nms in wts.files])

    model.set_ka(1e5)
    model.evaluate(cifar10.x_val,cifar10.y_val,batch_size=trr.val_bs,verbose=2)
    model.evaluate(cifar10.x_test,cifar10.y_test,batch_size=trr.val_bs,verbose=2)

    x_trains=cifar10.x_train[:]
    for _ in range(5):
        x_trains=x_trains[np.random.permutation(len(x_trains))]
        for ik in range(0,len(x_trains),trr.val_bs):
            model(x_trains[ik:ik+trr.val_bs],training=True)
            if (ik % 2000 == 0):
                print("#", end="")
        model.evaluate(cifar10.x_val,cifar10.y_val,batch_size=trr.val_bs,verbose=2)
        model.evaluate(cifar10.x_test,cifar10.y_test,batch_size=trr.val_bs,verbose=2)
