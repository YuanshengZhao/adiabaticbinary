import tensorflow as tf
import numpy as np
import layers
import cifar10
import trainer
import sys


mtp=1
class ResNet20(tf.keras.Model):
    def __init__(self,binW,binA):
        super(ResNet20, self).__init__()
        self.binW=binW
        self.binA=binA
        self.conv01 = tf.keras.layers.Conv2D(16*mtp,3,strides=1,padding='same',kernel_regularizer=tf.keras.regularizers.L2(2e-4))
        self.conv02 = layers.BinaryConv2DCL(16*mtp,3,1,False) if binW else tf.keras.layers.Conv2D(16*mtp,3,strides=1,padding='same', kernel_regularizer=tf.keras.regularizers.L2(2e-4))
        self.conv03 = layers.BinaryConv2DCL(16*mtp,3,1,False) if binW else tf.keras.layers.Conv2D(16*mtp,3,strides=1,padding='same', kernel_regularizer=tf.keras.regularizers.L2(2e-4))
        self.conv04 = layers.BinaryConv2DCL(16*mtp,3,1,False) if binW else tf.keras.layers.Conv2D(16*mtp,3,strides=1,padding='same', kernel_regularizer=tf.keras.regularizers.L2(2e-4))
        self.conv05 = layers.BinaryConv2DCL(16*mtp,3,1,False) if binW else tf.keras.layers.Conv2D(16*mtp,3,strides=1,padding='same', kernel_regularizer=tf.keras.regularizers.L2(2e-4))
        self.conv06 = layers.BinaryConv2DCL(16*mtp,3,1,False) if binW else tf.keras.layers.Conv2D(16*mtp,3,strides=1,padding='same', kernel_regularizer=tf.keras.regularizers.L2(2e-4))
        self.conv07 = layers.BinaryConv2DCL(16*mtp,3,1,False) if binW else tf.keras.layers.Conv2D(16*mtp,3,strides=1,padding='same', kernel_regularizer=tf.keras.regularizers.L2(2e-4))
        self.conv08 = layers.BinaryConv2DCL(32*mtp,3,2,False) if binW else tf.keras.layers.Conv2D(32*mtp,3,strides=2,padding='same', kernel_regularizer=tf.keras.regularizers.L2(2e-4))
        self.conv09 = layers.BinaryConv2DCL(32*mtp,3,1,False) if binW else tf.keras.layers.Conv2D(32*mtp,3,strides=1,padding='same', kernel_regularizer=tf.keras.regularizers.L2(2e-4))
        self.conv10 = layers.BinaryConv2DCL(32*mtp,3,1,False) if binW else tf.keras.layers.Conv2D(32*mtp,3,strides=1,padding='same', kernel_regularizer=tf.keras.regularizers.L2(2e-4))
        self.conv11 = layers.BinaryConv2DCL(32*mtp,3,1,False) if binW else tf.keras.layers.Conv2D(32*mtp,3,strides=1,padding='same', kernel_regularizer=tf.keras.regularizers.L2(2e-4))
        self.conv12 = layers.BinaryConv2DCL(32*mtp,3,1,False) if binW else tf.keras.layers.Conv2D(32*mtp,3,strides=1,padding='same', kernel_regularizer=tf.keras.regularizers.L2(2e-4))
        self.conv13 = layers.BinaryConv2DCL(32*mtp,3,1,False) if binW else tf.keras.layers.Conv2D(32*mtp,3,strides=1,padding='same', kernel_regularizer=tf.keras.regularizers.L2(2e-4))
        self.conv14 = layers.BinaryConv2DCL(64*mtp,3,2,False) if binW else tf.keras.layers.Conv2D(64*mtp,3,strides=2,padding='same', kernel_regularizer=tf.keras.regularizers.L2(2e-4))
        self.conv15 = layers.BinaryConv2DCL(64*mtp,3,1,False) if binW else tf.keras.layers.Conv2D(64*mtp,3,strides=1,padding='same', kernel_regularizer=tf.keras.regularizers.L2(2e-4))
        self.conv16 = layers.BinaryConv2DCL(64*mtp,3,1,False) if binW else tf.keras.layers.Conv2D(64*mtp,3,strides=1,padding='same', kernel_regularizer=tf.keras.regularizers.L2(2e-4))
        self.conv17 = layers.BinaryConv2DCL(64*mtp,3,1,False) if binW else tf.keras.layers.Conv2D(64*mtp,3,strides=1,padding='same', kernel_regularizer=tf.keras.regularizers.L2(2e-4))
        self.conv18 = layers.BinaryConv2DCL(64*mtp,3,1,False) if binW else tf.keras.layers.Conv2D(64*mtp,3,strides=1,padding='same', kernel_regularizer=tf.keras.regularizers.L2(2e-4))
        self.conv19 = layers.BinaryConv2DCL(64*mtp,3,1,False) if binW else tf.keras.layers.Conv2D(64*mtp,3,strides=1,padding='same', kernel_regularizer=tf.keras.regularizers.L2(2e-4))
        self.globalavgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense = tf.keras.layers.Dense(10,activation='softmax',kernel_regularizer=tf.keras.regularizers.l2(2e-4))
        self.actv01=layers.BinaryActivation(True) if binA else tf.keras.layers.Activation("relu")
        self.actv03=layers.BinaryActivation(True) if binA else tf.keras.layers.Activation("relu")
        self.actv05=layers.BinaryActivation(True) if binA else tf.keras.layers.Activation("relu")
        self.actv07=layers.BinaryActivation(True) if binA else tf.keras.layers.Activation("relu")
        self.actv09=layers.BinaryActivation(True) if binA else tf.keras.layers.Activation("relu")
        self.actv11=layers.BinaryActivation(True) if binA else tf.keras.layers.Activation("relu")
        self.actv13=layers.BinaryActivation(True) if binA else tf.keras.layers.Activation("relu")
        self.actv15=layers.BinaryActivation(True) if binA else tf.keras.layers.Activation("relu")
        self.actv17=layers.BinaryActivation(True) if binA else tf.keras.layers.Activation("relu")
        self.actv02=layers.BinaryActivation(True) if binA else tf.keras.layers.Activation("relu")
        self.actv04=layers.BinaryActivation(True) if binA else tf.keras.layers.Activation("relu")
        self.actv06=layers.BinaryActivation(True) if binA else tf.keras.layers.Activation("relu")
        self.actv08=layers.BinaryActivation(True) if binA else tf.keras.layers.Activation("relu")
        self.actv10=layers.BinaryActivation(True) if binA else tf.keras.layers.Activation("relu")
        self.actv12=layers.BinaryActivation(True) if binA else tf.keras.layers.Activation("relu")
        self.actv14=layers.BinaryActivation(True) if binA else tf.keras.layers.Activation("relu")
        self.actv16=layers.BinaryActivation(True) if binA else tf.keras.layers.Activation("relu")
        self.actv18=layers.BinaryActivation(True) if binA else tf.keras.layers.Activation("relu")
        self.actv19=tf.keras.layers.Activation('relu')
        self.batnor01=tf.keras.layers.BatchNormalization(epsilon=.01)
        self.batnor02=tf.keras.layers.BatchNormalization(epsilon=.01)
        self.batnor03=tf.keras.layers.BatchNormalization(epsilon=.01)
        self.batnor04=tf.keras.layers.BatchNormalization(epsilon=.01)
        self.batnor05=tf.keras.layers.BatchNormalization(epsilon=.01)
        self.batnor06=tf.keras.layers.BatchNormalization(epsilon=.01)
        self.batnor07=tf.keras.layers.BatchNormalization(epsilon=.01)
        self.batnor08=tf.keras.layers.BatchNormalization(epsilon=.01)
        self.batnor09=tf.keras.layers.BatchNormalization(epsilon=.01)
        self.batnor10=tf.keras.layers.BatchNormalization(epsilon=.01)
        self.batnor11=tf.keras.layers.BatchNormalization(epsilon=.01)
        self.batnor12=tf.keras.layers.BatchNormalization(epsilon=.01)
        self.batnor13=tf.keras.layers.BatchNormalization(epsilon=.01)
        self.batnor14=tf.keras.layers.BatchNormalization(epsilon=.01)
        self.batnor15=tf.keras.layers.BatchNormalization(epsilon=.01)
        self.batnor16=tf.keras.layers.BatchNormalization(epsilon=.01)
        self.batnor17=tf.keras.layers.BatchNormalization(epsilon=.01)
        self.batnor18=tf.keras.layers.BatchNormalization(epsilon=.01)
        self.batnor19=tf.keras.layers.BatchNormalization(epsilon=.01)

    def set_kk(self,kkv):
        if not self.binW:
            raise NotImplementedError
        self.conv02.set_kk(kkv)
        self.conv04.set_kk(kkv)
        self.conv06.set_kk(kkv)
        self.conv08.set_kk(kkv)
        self.conv10.set_kk(kkv)
        self.conv12.set_kk(kkv)
        self.conv14.set_kk(kkv)
        self.conv16.set_kk(kkv)
        self.conv18.set_kk(kkv)
        self.conv03.set_kk(kkv)
        self.conv05.set_kk(kkv)
        self.conv07.set_kk(kkv)
        self.conv09.set_kk(kkv)
        self.conv11.set_kk(kkv)
        self.conv13.set_kk(kkv)
        self.conv15.set_kk(kkv)
        self.conv17.set_kk(kkv)
        self.conv19.set_kk(kkv)

    def set_ka(self,kkv):
        if not self.binA:
            raise NotImplementedError
        self.actv02.set_kk(kkv)
        self.actv04.set_kk(kkv)
        self.actv06.set_kk(kkv)
        self.actv08.set_kk(kkv)
        self.actv10.set_kk(kkv)
        self.actv12.set_kk(kkv)
        self.actv14.set_kk(kkv)
        self.actv16.set_kk(kkv)
        self.actv18.set_kk(kkv)
        self.actv01.set_kk(kkv)
        self.actv03.set_kk(kkv)
        self.actv05.set_kk(kkv)
        self.actv07.set_kk(kkv)
        self.actv09.set_kk(kkv)
        self.actv11.set_kk(kkv)
        self.actv13.set_kk(kkv)
        self.actv15.set_kk(kkv)
        self.actv17.set_kk(kkv)


    def get_kk(self):
        if not self.binW:
            raise NotImplementedError
        return self.conv02.kk

    def get_ka(self):
        if not self.binA:
            raise NotImplementedError
        return self.actv02.kk

    def call(self, inputs, training=False):
        x = self.batnor01(self.conv01(inputs),training=training)
        x=x+self.conv03(self.actv02(self.batnor02(self.conv02(self.actv01(x)),training=training)))
        x=x+self.conv05(self.actv04(self.batnor04(self.conv04(self.actv03(self.batnor03(x,training=training))),training=training)))
        x=x+self.conv07(self.actv06(self.batnor06(self.conv06(self.actv05(self.batnor05(x,training=training))),training=training)))
        x=tf.pad(tf.nn.avg_pool2d(x,2,strides=2, padding='SAME'),[[0,0],[0,0],[0,0],[8,8]])+self.conv09(self.actv08(self.batnor08(self.conv08(self.actv07(self.batnor07(x,training=training))),training=training)))
        x=x+self.conv11(self.actv10(self.batnor10(self.conv10(self.actv09(self.batnor09(x,training=training))),training=training)))
        x=x+self.conv13(self.actv12(self.batnor12(self.conv12(self.actv11(self.batnor11(x,training=training))),training=training)))
        x=tf.pad(tf.nn.avg_pool2d(x,2,strides=2, padding='SAME'),[[0,0],[0,0],[0,0],[16,16]])+self.conv15(self.actv14(self.batnor14(self.conv14(self.actv13(self.batnor13(x,training=training))),training=training)))
        x=x+self.conv17(self.actv16(self.batnor16(self.conv16(self.actv15(self.batnor15(x,training=training))),training=training)))
        x=x+self.conv19(self.actv18(self.batnor18(self.conv18(self.actv17(self.batnor17(x,training=training))),training=training)))
        x=self.actv19(self.batnor19(x))
        return self.dense(self.globalavgpool(x))

class ResNet20_l2EVAL(ResNet20):
    def __init__(self,binW,binA):
        super().__init__(binW,binA)
    def call(self, inputs, training=False):
        x = self.actv01(self.batnor01(self.conv01(inputs),training=training))
        x=x+self.conv03(self.actv02(self.batnor02(self.conv02(x),training=training)))
        x=x+self.conv05(self.actv04(self.batnor04(self.conv04(self.actv03(self.batnor03(x,training=training))),training=training)))
        x=x+self.conv07(self.actv06(self.batnor06(self.conv06(self.actv05(self.batnor05(x,training=training))),training=training)))
        x=tf.pad(tf.nn.avg_pool2d(x,2,strides=2, padding='SAME'),[[0,0],[0,0],[0,0],[8,8]])+self.conv09(self.actv08(self.batnor08(self.conv08(self.actv07(self.batnor07(x,training=training))),training=training)))
        x=x+self.conv11(self.actv10(self.batnor10(self.conv10(self.actv09(self.batnor09(x,training=training))),training=training)))
        x=x+self.conv13(self.actv12(self.batnor12(self.conv12(self.actv11(self.batnor11(x,training=training))),training=training)))
        x=tf.pad(tf.nn.avg_pool2d(x,2,strides=2, padding='SAME'),[[0,0],[0,0],[0,0],[16,16]])+self.conv15(self.actv14(self.batnor14(self.conv14(self.actv13(self.batnor13(x,training=training))),training=training)))
        x=x+self.conv17(self.actv16(self.batnor16(self.conv16(self.actv15(self.batnor15(x,training=training))),training=training)))
        x=x+self.conv19(self.actv18(self.batnor18(self.conv18(self.actv17(self.batnor17(x,training=training))),training=training)))
        x=self.actv19(self.batnor19(x))
        return self.dense(self.globalavgpool(x))

mode=sys.argv[1]
rn=int(sys.argv[2])
if(mode=="w"):
    bw=1
    ba=0
else:
    print("mode = w|a|b")
    quit()

print("ResNet20, mode "+mode)


if(bw==1 and ba==0):
    model=ResNet20(0,0)
    optz=tf.keras.optimizers.SGD(learning_rate=.1,momentum=0.9,nesterov=True,clipvalue=3.0)
    model(cifar10.x_train[:2])
    model.compile(optimizer=optz,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    model.summary()
    model.evaluate(cifar10.x_val,cifar10.y_val,batch_size=125,verbose=2)
    model.fit(cifar10.datagen.flow(cifar10.x_train,cifar10.y_train,batch_size=100),epochs=80,validation_data=(cifar10.x_val,cifar10.y_val),verbose=2)
    np.savez("DResNet20_l2fp1.npz",*(model.get_weights()))
    optz.learning_rate.assign(1e-2)
    model.fit(cifar10.datagen.flow(cifar10.x_train,cifar10.y_train,batch_size=100),epochs=40,validation_data=(cifar10.x_val,cifar10.y_val),verbose=2)
    np.savez("DResNet20_l2fp2.npz",*(model.get_weights()))
    optz.learning_rate.assign(1e-3)
    model.fit(cifar10.datagen.flow(cifar10.x_train,cifar10.y_train,batch_size=100),epochs=20,validation_data=(cifar10.x_val,cifar10.y_val),verbose=2)
    np.savez("DResNet20_l2fp3.npz",*(model.get_weights()))

    with np.load("DResNet20_l2fp3.npz") as wts:
        wtt=[wts[nms] for nms in wts.files]

    for k in [37,35,33,31,29,27,25,23,21,19,17,15,13,11,9,7,5,3]: #nmk
        wtt.insert(k,np.array(tf.reduce_max(tf.math.abs(wtt[k-1])).numpy()*1.5))
    for k in [2,5,8,11,14,17,20,23,26,29,32,35,38,41,44,47,50,53]: # reset kernel
        wtt[k]=tf.math.atanh(wtt[k]/tf.reduce_max(tf.math.abs(wtt[k]))/1.5).numpy()
    for k in [1,4,7,10,13,16,19,22,25,28,31,34,37,40,43,46,49,52,55]: # remove bias
        wtt[k]*=0
    for k in [56,53,50,47,44,41,38,35,32,29,26,23,20,17,14,11,8,5]: # kk
        wtt.insert(k,np.array(1.0))

    model=ResNet20(1,0)
    optz=tf.keras.optimizers.SGD(learning_rate=.01,momentum=0.9,nesterov=True,clipvalue=3.0)
    model(cifar10.x_train[:2])
    model.compile(optimizer=optz,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    model.summary()
    model.set_weights(wtt)
    model.evaluate(cifar10.x_val,cifar10.y_val,batch_size=125,verbose=2)

    trr=trainer.Trainer(50,"w")
    trr.prto=1.2
    trr.maxpush=3
    trr.target_acc=.9 # higher value also works but is slower
    trr.lr_power=0
    trr.train(model,cifar10.datagen,cifar10.x_train,cifar10.y_train,cifar10.x_val,cifar10.y_val,600,1e-2,"DResNet20_l2"+str(rn),True,cifar10.x_test,cifar10.y_test)

    with np.load("DResNet20_l2"+str(rn) + mode + "_Best.npz") as wts:
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
