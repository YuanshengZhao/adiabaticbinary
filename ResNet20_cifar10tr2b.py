import tensorflow as tf
import numpy as np
import layers
import cifar10
import trainer2 as trainer
import sys

mtp=1
class ResNet20(tf.keras.Model):
    def __init__(self,binW,binA):
        super(ResNet20, self).__init__()
        self.binW=binW
        self.binA=binA
        self.conv01 = tf.keras.layers.Conv2D(16*mtp,3,strides=1,padding='same')
        self.conv02 = layers.BinaryConv2D(16*mtp,3,1,False) if binW else tf.keras.layers.Conv2D(16*mtp,3,strides=1,padding='same')
        self.conv03 = layers.BinaryConv2D(16*mtp,3,1,False) if binW else tf.keras.layers.Conv2D(16*mtp,3,strides=1,padding='same')
        self.conv04 = layers.BinaryConv2D(16*mtp,3,1,False) if binW else tf.keras.layers.Conv2D(16*mtp,3,strides=1,padding='same')
        self.conv05 = layers.BinaryConv2D(16*mtp,3,1,False) if binW else tf.keras.layers.Conv2D(16*mtp,3,strides=1,padding='same')
        self.conv06 = layers.BinaryConv2D(16*mtp,3,1,False) if binW else tf.keras.layers.Conv2D(16*mtp,3,strides=1,padding='same')
        self.conv07 = layers.BinaryConv2D(16*mtp,3,1,False) if binW else tf.keras.layers.Conv2D(16*mtp,3,strides=1,padding='same')
        self.conv08 = layers.BinaryConv2D(32*mtp,3,2,False) if binW else tf.keras.layers.Conv2D(32*mtp,3,strides=2,padding='same')
        self.conv09 = layers.BinaryConv2D(32*mtp,3,1,False) if binW else tf.keras.layers.Conv2D(32*mtp,3,strides=1,padding='same')
        self.conv10 = layers.BinaryConv2D(32*mtp,3,1,False) if binW else tf.keras.layers.Conv2D(32*mtp,3,strides=1,padding='same')
        self.conv11 = layers.BinaryConv2D(32*mtp,3,1,False) if binW else tf.keras.layers.Conv2D(32*mtp,3,strides=1,padding='same')
        self.conv12 = layers.BinaryConv2D(32*mtp,3,1,False) if binW else tf.keras.layers.Conv2D(32*mtp,3,strides=1,padding='same')
        self.conv13 = layers.BinaryConv2D(32*mtp,3,1,False) if binW else tf.keras.layers.Conv2D(32*mtp,3,strides=1,padding='same')
        self.conv14 = layers.BinaryConv2D(64*mtp,3,2,False) if binW else tf.keras.layers.Conv2D(64*mtp,3,strides=2,padding='same')
        self.conv15 = layers.BinaryConv2D(64*mtp,3,1,False) if binW else tf.keras.layers.Conv2D(64*mtp,3,strides=1,padding='same')
        self.conv16 = layers.BinaryConv2D(64*mtp,3,1,False) if binW else tf.keras.layers.Conv2D(64*mtp,3,strides=1,padding='same')
        self.conv17 = layers.BinaryConv2D(64*mtp,3,1,False) if binW else tf.keras.layers.Conv2D(64*mtp,3,strides=1,padding='same')
        self.conv18 = layers.BinaryConv2D(64*mtp,3,1,False) if binW else tf.keras.layers.Conv2D(64*mtp,3,strides=1,padding='same')
        self.conv19 = layers.BinaryConv2D(64*mtp,3,1,False) if binW else tf.keras.layers.Conv2D(64*mtp,3,strides=1,padding='same')
        self.convd1 = layers.BinaryConv2D(32*mtp,1,2,False) if binW else tf.keras.layers.Conv2D(32*mtp,1,strides=2,padding='same')
        self.convd2 = layers.BinaryConv2D(64*mtp,1,2,False) if binW else tf.keras.layers.Conv2D(64*mtp,1,strides=2,padding='same')
        self.globalavgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense = tf.keras.layers.Dense(10,activation='softmax',kernel_regularizer=tf.keras.regularizers.l2(2e-4))
        # BinaryActivationHT(False), BinaryActivationRL(False) also works.
        self.actv01=layers.BinaryActivation(False) if binA else tf.keras.layers.Activation("relu")
        self.actv03=layers.BinaryActivation(False) if binA else tf.keras.layers.Activation("relu")
        self.actv05=layers.BinaryActivation(False) if binA else tf.keras.layers.Activation("relu")
        self.actv07=layers.BinaryActivation(False) if binA else tf.keras.layers.Activation("relu")
        self.actv09=layers.BinaryActivation(False) if binA else tf.keras.layers.Activation("relu")
        self.actv11=layers.BinaryActivation(False) if binA else tf.keras.layers.Activation("relu")
        self.actv13=layers.BinaryActivation(False) if binA else tf.keras.layers.Activation("relu")
        self.actv15=layers.BinaryActivation(False) if binA else tf.keras.layers.Activation("relu")
        self.actv17=layers.BinaryActivation(False) if binA else tf.keras.layers.Activation("relu")
        self.actv02=layers.BinaryActivation(False) if binA else tf.keras.layers.Activation("relu")
        self.actv04=layers.BinaryActivation(False) if binA else tf.keras.layers.Activation("relu")
        self.actv06=layers.BinaryActivation(False) if binA else tf.keras.layers.Activation("relu")
        self.actv08=layers.BinaryActivation(False) if binA else tf.keras.layers.Activation("relu")
        self.actv10=layers.BinaryActivation(False) if binA else tf.keras.layers.Activation("relu")
        self.actv12=layers.BinaryActivation(False) if binA else tf.keras.layers.Activation("relu")
        self.actv14=layers.BinaryActivation(False) if binA else tf.keras.layers.Activation("relu")
        self.actv16=layers.BinaryActivation(False) if binA else tf.keras.layers.Activation("relu")
        self.actv18=layers.BinaryActivation(False) if binA else tf.keras.layers.Activation("relu")
        self.actv19=tf.keras.layers.Activation('relu')
        self.batnor01=tf.keras.layers.BatchNormalization()
        self.batnor02=tf.keras.layers.BatchNormalization()
        self.batnor03=tf.keras.layers.BatchNormalization()
        self.batnor04=tf.keras.layers.BatchNormalization()
        self.batnor05=tf.keras.layers.BatchNormalization()
        self.batnor06=tf.keras.layers.BatchNormalization()
        self.batnor07=tf.keras.layers.BatchNormalization()
        self.batnor08=tf.keras.layers.BatchNormalization()
        self.batnor09=tf.keras.layers.BatchNormalization()
        self.batnor10=tf.keras.layers.BatchNormalization()
        self.batnor11=tf.keras.layers.BatchNormalization()
        self.batnor12=tf.keras.layers.BatchNormalization()
        self.batnor13=tf.keras.layers.BatchNormalization()
        self.batnor14=tf.keras.layers.BatchNormalization()
        self.batnor15=tf.keras.layers.BatchNormalization()
        self.batnor16=tf.keras.layers.BatchNormalization()
        self.batnor17=tf.keras.layers.BatchNormalization()
        self.batnor18=tf.keras.layers.BatchNormalization()
        self.batnor19=tf.keras.layers.BatchNormalization()
        self.batnoh03=tf.keras.layers.BatchNormalization()
        self.batnoh01=tf.keras.layers.BatchNormalization()
        self.batnoh05=tf.keras.layers.BatchNormalization()
        self.batnoh07=tf.keras.layers.BatchNormalization()
        self.batnoh09=tf.keras.layers.BatchNormalization()
        self.batnoh11=tf.keras.layers.BatchNormalization()
        self.batnoh13=tf.keras.layers.BatchNormalization()
        self.batnoh15=tf.keras.layers.BatchNormalization()
        self.batnoh17=tf.keras.layers.BatchNormalization()
        self.drop01=tf.keras.layers.Dropout(.1)
        self.drop02=tf.keras.layers.Dropout(.1)
        self.drop03=tf.keras.layers.Dropout(.1)
        self.drop04=tf.keras.layers.Dropout(.1)
        self.drop05=tf.keras.layers.Dropout(.1)
        self.drop06=tf.keras.layers.Dropout(.1)
        self.drop07=tf.keras.layers.Dropout(.1)
        self.drop08=tf.keras.layers.Dropout(.1)
        self.drop09=tf.keras.layers.Dropout(.1)
        self.drop10=tf.keras.layers.Dropout(.1)
        self.drop11=tf.keras.layers.Dropout(.1)
        self.drop12=tf.keras.layers.Dropout(.1)
        self.drop13=tf.keras.layers.Dropout(.1)
        self.drop14=tf.keras.layers.Dropout(.1)
        self.drop15=tf.keras.layers.Dropout(.1)
        self.drop16=tf.keras.layers.Dropout(.1)
        self.drop17=tf.keras.layers.Dropout(.1)
        self.drop18=tf.keras.layers.Dropout(.1)
        self.drop19=tf.keras.layers.Dropout(.1)

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
        x=               self.batnor01(self.conv01(inputs),training=True)
        x=x    +         self.batnor03(self.conv03(self.drop02(self.actv02(self.batnor02(self.conv02(self.drop01(self.actv01(self.batnoh01(x,training=True)),training=training)),training=True)),training=training)),training=True)
        x=x    +         self.batnor05(self.conv05(self.drop04(self.actv04(self.batnor04(self.conv04(self.drop03(self.actv03(self.batnoh03(x,training=True)),training=training)),training=True)),training=training)),training=True)
        x=x    +         self.batnor07(self.conv07(self.drop06(self.actv06(self.batnor06(self.conv06(self.drop05(self.actv05(self.batnoh05(x,training=True)),training=training)),training=True)),training=training)),training=True)
        x=self.convd1(x)+self.batnor09(self.conv09(self.drop08(self.actv08(self.batnor08(self.conv08(self.drop07(self.actv07(self.batnoh07(x,training=True)),training=training)),training=True)),training=training)),training=True)
        x=x    +         self.batnor11(self.conv11(self.drop10(self.actv10(self.batnor10(self.conv10(self.drop09(self.actv09(self.batnoh09(x,training=True)),training=training)),training=True)),training=training)),training=True)
        x=x    +         self.batnor13(self.conv13(self.drop12(self.actv12(self.batnor12(self.conv12(self.drop11(self.actv11(self.batnoh11(x,training=True)),training=training)),training=True)),training=training)),training=True)
        x=self.convd2(x)+self.batnor15(self.conv15(self.drop14(self.actv14(self.batnor14(self.conv14(self.drop13(self.actv13(self.batnoh13(x,training=True)),training=training)),training=True)),training=training)),training=True)
        x=x    +         self.batnor17(self.conv17(self.drop16(self.actv16(self.batnor16(self.conv16(self.drop15(self.actv15(self.batnoh15(x,training=True)),training=training)),training=True)),training=training)),training=True)
        x=x    +         self.batnor19(self.conv19(self.drop18(self.actv18(self.batnor18(self.conv18(self.drop17(self.actv17(self.batnoh17(x,training=True)),training=training)),training=True)),training=training)),training=True)
        x=self.drop19(self.actv19(x),training=training)
        return self.dense(self.globalavgpool(x))

class ResNet20_EVAL(ResNet20):
    def call(self, inputs, training=False):
        x=               self.batnor01(self.conv01(inputs),training=training)
        x=x    +         self.batnor03(self.conv03(self.actv02(self.batnor02(self.conv02(self.actv01(self.batnoh01(x,training=training))),training=training))),training=training)
        x=x    +         self.batnor05(self.conv05(self.actv04(self.batnor04(self.conv04(self.actv03(self.batnoh03(x,training=training))),training=training))),training=training)
        x=x    +         self.batnor07(self.conv07(self.actv06(self.batnor06(self.conv06(self.actv05(self.batnoh05(x,training=training))),training=training))),training=training)
        x=self.convd1(x)+self.batnor09(self.conv09(self.actv08(self.batnor08(self.conv08(self.actv07(self.batnoh07(x,training=training))),training=training))),training=training)
        x=x    +         self.batnor11(self.conv11(self.actv10(self.batnor10(self.conv10(self.actv09(self.batnoh09(x,training=training))),training=training))),training=training)
        x=x    +         self.batnor13(self.conv13(self.actv12(self.batnor12(self.conv12(self.actv11(self.batnoh11(x,training=training))),training=training))),training=training)
        x=self.convd2(x)+self.batnor15(self.conv15(self.actv14(self.batnor14(self.conv14(self.actv13(self.batnoh13(x,training=training))),training=training))),training=training)
        x=x    +         self.batnor17(self.conv17(self.actv16(self.batnor16(self.conv16(self.actv15(self.batnoh15(x,training=training))),training=training))),training=training)
        x=x    +         self.batnor19(self.conv19(self.actv18(self.batnor18(self.conv18(self.actv17(self.batnoh17(x,training=training))),training=training))),training=training)
        x=self.actv19(x)
        return self.dense(self.globalavgpool(x))

mode=sys.argv[1]
rn=int(sys.argv[2])

trr=trainer.Trainer(15,mode)
trr.prto=1.5
trr.maxpush=1
trr.lr_base=1.0
trr.val_bs=125
trr.reduction=0.6

if(mode=="a"):
    bw=0
    ba=1
    init_lr=.01
    optz=tf.keras.optimizers.Adam(clipvalue=.01)
    # init_lr=.1
    # optz=tf.keras.optimizers.SGD(learning_rate=init_lr,momentum=0.9,nesterov=True,clipvalue=.01)
    trr.lr_power=1
elif(mode=="b"):
    bw=1
    ba=1
    init_lr=.01
    optz=tf.keras.optimizers.Adam(clipvalue=.01)
    # init_lr=.1
    # optz=tf.keras.optimizers.SGD(learning_rate=init_lr,momentum=0.9,nesterov=True,clipvalue=.01)
    trr.lr_power=1
elif(mode=="w"):
    bw=1
    ba=0
    trr.lr_power=0
    init_lr=.1
    optz=tf.keras.optimizers.SGD(learning_rate=init_lr,momentum=0.9,nesterov=True,clipvalue=.3)
else:
    print("mode = w|a|b")
    quit()

print("ResNet20, mode "+mode)

model=ResNet20(bw,ba)
model(cifar10.x_train[:2])
model.compile(optimizer=optz,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# remove this to train full-binary from scratch
if(ba and bw):
    with np.load("DResNet20tr2b_1a_Best.npz") as wts:
        wtt=[wts[nms] for nms in wts.files]
    for k in [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41]:
        wtt[k]*=0
    for k in [41,39,37,35,33,31,29,27,25,23,21,19,17,15,13,11,9,7,5,3]:
        wtt.insert(k,np.array(1.0))
    for k in [62,59,56,53,50,47,44,41,38,35,32,29,26,23,20,17,14,11,8,5]:
        wtt.insert(k,np.array(1.0))

    model.set_weights(wtt)
if(bw):
    model.set_kk(trr.lr_base)
if(ba):
    model.set_ka(trr.lr_base)

model.evaluate(cifar10.x_val,cifar10.y_val,batch_size=trr.val_bs,verbose=2)

trr.sw_epc=0
trr.train(model,cifar10.datagen,cifar10.x_train,cifar10.y_train,cifar10.x_val,cifar10.y_val,80,init_lr,"DResNet20tr2b_"+str(rn),True,cifar10.x_test,cifar10.y_test)
trr.train(model,cifar10.datagen,cifar10.x_train,cifar10.y_train,cifar10.x_val,cifar10.y_val,600,init_lr/10,"DResNet20tr2b_"+str(rn),True,cifar10.x_test,cifar10.y_test)

modelE=ResNet20_EVAL(bw,ba)
modelE(cifar10.x_train[:2])
modelE.compile(optimizer=optz,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

with np.load("DResNet20tr2b_"+str(rn) + mode + "_Best.npz") as wts:
    modelE.set_weights([wts[nms] for nms in wts.files])

if(bw):
    modelE.set_kk(1e5)
if(ba):
    modelE.set_ka(1e5)

modelE.evaluate(cifar10.x_val,cifar10.y_val,batch_size=trr.val_bs,verbose=2)
x_trains=cifar10.x_train[:]
for _ in range(10):
    x_trains=x_trains[np.random.permutation(len(x_trains))]
    for ik in range(0,len(x_trains),trr.val_bs*2):
        modelE(x_trains[ik:ik+trr.val_bs*2],training=True)
        if (ik % 2000 == 0):
            print("#", end="")
    modelE.evaluate(cifar10.x_val,cifar10.y_val,batch_size=trr.val_bs,verbose=2)
    modelE.evaluate(cifar10.x_test,cifar10.y_test,batch_size=trr.val_bs,verbose=2)