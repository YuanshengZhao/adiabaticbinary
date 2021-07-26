import tensorflow as tf

class BinaryConv2D(tf.keras.layers.Layer):
    def __init__(self, num_chanel, ker_size=3, num_stride=1, ker_bias=False):
        super(BinaryConv2D, self).__init__()
        self.num_chanel = num_chanel
        self.ker_size = ker_size
        self.num_stride = num_stride
        self.ker_bias = ker_bias

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=[self.ker_size,self.ker_size,int(input_shape[-1]),self.num_chanel],
                                        initializer=tf.keras.initializers.RandomUniform(minval=-.1, maxval=.1))
        self.nmk = self.add_weight(initializer=tf.keras.initializers.Constant(1.))
        self.bias = self.add_weight(trainable=self.ker_bias, shape=[self.num_chanel],initializer=tf.keras.initializers.Constant(0.))
        self.kk = self.add_weight(trainable=False,initializer=tf.keras.initializers.Constant(1.))

    def set_kk(self,kknew):
        self.kk.assign(kknew)

    def call(self, inputs):
        if (self.kk < 1e3):
            return self.nmk * tf.nn.conv2d(inputs, tf.math.tanh(self.kernel*self.kk)+self.bias, self.num_stride, "SAME")
        else:
            return self.nmk * tf.nn.conv2d(inputs, tf.math.sign(self.kernel)        +self.bias, self.num_stride, "SAME")


class BinaryConv1D(tf.keras.layers.Layer):
    def __init__(self, num_chanel, ker_size=3, num_stride=1, ker_bias=False):
        super(BinaryConv1D, self).__init__()
        self.num_chanel = num_chanel
        self.ker_size = ker_size
        self.num_stride = num_stride
        self.ker_bias = ker_bias

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=[self.ker_size,int(input_shape[-1]),self.num_chanel],
                                        initializer=tf.keras.initializers.RandomUniform(minval=-.1, maxval=.1))
        self.nmk = self.add_weight(initializer=tf.keras.initializers.Constant(1.))
        self.bias = self.add_weight(trainable=self.ker_bias, shape=[self.num_chanel],initializer=tf.keras.initializers.Constant(0.))
        self.kk = self.add_weight(trainable=False,initializer=tf.keras.initializers.Constant(1.))

    def set_kk(self,kknew):
        self.kk.assign(kknew)

    def call(self, inputs):
        if (self.kk < 1e3):
            return self.nmk * tf.nn.conv1d(inputs, tf.math.tanh(self.kernel*self.kk)+self.bias, self.num_stride, "SAME")
        else:
            return self.nmk * tf.nn.conv1d(inputs, tf.math.sign(self.kernel)        +self.bias, self.num_stride, "SAME")


class BinaryConv2DCL(BinaryConv2D):
    def call(self, inputs):
        self.add_loss(2e-1*(tf.math.reduce_sum(tf.nn.relu(tf.math.abs(self.kernel)-.2)**2)))
        if (self.kk < 1e3):
            return self.nmk * tf.nn.conv2d(inputs, tf.math.tanh(self.kernel*self.kk)+self.bias, self.num_stride, "SAME")
        else:
            return self.nmk * tf.nn.conv2d(inputs, tf.math.sign(self.kernel)        +self.bias, self.num_stride, "SAME")


class BinaryDense(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(BinaryDense, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=[int(input_shape[-1]),self.num_outputs],initializer=tf.keras.initializers.RandomUniform(minval=-.1,maxval=.1))
        self.bias = self.add_weight(shape=[self.num_outputs],initializer=tf.keras.initializers.Constant(0.))
        self.nmk = self.add_weight(initializer=tf.keras.initializers.Constant(1.))
        self.kk = self.add_weight(trainable=False,initializer=tf.keras.initializers.Constant(1.))

    def set_kk(self,kknew):
        self.kk.assign(kknew)

    def call(self, inputs):
        if(self.kk < 1e3):
            return self.nmk * tf.matmul(inputs, tf.math.tanh(self.kernel*self.kk))+self.bias
        else:
            return self.nmk * tf.matmul(inputs, tf.math.sign(self.kernel))        +self.bias


class BinaryActivation(tf.keras.layers.Layer):
    def __init__(self, ker_bias=False):
        super(BinaryActivation, self).__init__()
        self.ker_bias = ker_bias

    def build(self, input_shape):
        self.bias = self.add_weight(trainable=self.ker_bias,initializer=tf.keras.initializers.Constant(1.0))
        self.kk = self.add_weight(trainable=False,initializer=tf.keras.initializers.Constant(1.0))

    def set_kk(self,kkx):
        self.kk.assign(kkx)

    def call(self, inputs):
        if(self.kk < 1e3):
            return tf.math.tanh(inputs*self.kk)+self.bias
        else:
            return tf.math.sign(inputs)+self.bias


class BinaryActivationH(BinaryActivation):
    def build(self, input_shape):
        self.bias = self.add_weight(trainable=self.ker_bias,initializer=tf.keras.initializers.Constant(0.0))
        self.kk = self.add_weight(trainable=False,initializer=tf.keras.initializers.Constant(1.0))
    def call(self, inputs):
        if(self.kk < 1e3):
            return tf.nn.sigmoid(inputs*self.kk)+self.bias
        else:
            return tf.math.sign(inputs)*.5+(.5+self.bias)
            
class BinaryActivationCLU(BinaryActivation):
    def build(self, input_shape):
        self.bias = self.add_weight(trainable=self.ker_bias,initializer=tf.keras.initializers.Constant(0.0))
        self.kk = self.add_weight(trainable=False,initializer=tf.keras.initializers.Constant(1.0))
    def call(self, inputs):
        if(self.kk < 1e3):
            return tf.clip_by_value(inputs*self.kk,0,1)+self.bias
        else:
            return tf.math.sign(inputs)*.5+(.5+self.bias)

class BinaryActivationHT(BinaryActivation):
    def call(self, inputs):
        if(self.kk < 1e3):
            return tf.clip_by_value(inputs*self.kk,-1,1)+self.bias
        else:
            return tf.math.sign(inputs)+self.bias

class BinaryActivationRL(BinaryActivation):
    def build(self, input_shape):
        self.bias = self.add_weight(trainable=self.ker_bias,initializer=tf.keras.initializers.Constant(0.0))
        self.kk = self.add_weight(trainable=False,initializer=tf.keras.initializers.Constant(1.0))
    def call(self, inputs):
        if(self.kk < 1e3):
            return tf.math.tanh(tf.nn.relu(inputs)*self.kk)+self.bias
        else:
            return tf.math.sign(inputs)/2+(.5+self.bias)
            
class BinaryActivationBS(tf.keras.layers.Layer):
    def __init__(self):
        super(BinaryActivationBS, self).__init__()

    def build(self, input_shape):
        self.bis2 = self.add_weight(initializer=tf.keras.initializers.Constant(0.0))
        self.kk = self.add_weight(trainable=False,initializer=tf.keras.initializers.Constant(1.0))
        self.bias = self.add_weight(trainable=False,shape=input_shape[1:],
                                    initializer=tf.keras.initializers.Constant(0.0))
        self.maxs = self.add_weight(trainable=False,shape=input_shape[1:],
                                    initializer=tf.keras.initializers.Constant(0.0))
        self.mins = self.add_weight(trainable=False,shape=input_shape[1:],
                                    initializer=tf.keras.initializers.Constant(0.0))
    def set_bias(self):
        self.bias.assign(tf.clip_by_value(self.bias,self.mins,self.maxs)*.1+self.bias*.9)
        # self.bias.assign(tf.clip_by_value(self.bias,
        #                                   self.mins+(self.maxs-self.mins)*5e-3*tf.random.uniform(shape=self.mins.shape),
        #                                   self.maxs-(self.maxs-self.mins)*5e-3*tf.random.uniform(shape=self.mins.shape))
        #                 )        
        self.mins.assign(self.bias+1e5)
        self.maxs.assign(self.bias-1e5)
    def set_kk(self,kknew):
        self.kk.assign(kknew)
    def call(self, inputs,training=False):
        if (training):
            self.maxs.assign(tf.reduce_max([tf.reduce_max(inputs,axis=0),self.maxs],axis=0))
            self.mins.assign(tf.reduce_min([tf.reduce_min(inputs,axis=0),self.mins],axis=0))
        if (self.kk < 1e3):
            return tf.math.sigmoid(self.kk*(inputs-self.bias))+self.bis2
        else:
            return tf.math.sign(inputs-self.bias)*.5+.5+self.bis2


class BinaryActivationP(tf.keras.layers.Layer):
    def __init__(self, ker_bias=False):
        super(BinaryActivationP, self).__init__()
        self.ker_bias = ker_bias

    def build(self, input_shape):
        self.bias = self.add_weight(trainable=self.ker_bias,initializer=tf.keras.initializers.Constant(0.0))
        self.kk = self.add_weight(trainable=False,initializer=tf.keras.initializers.Constant(1.0))


    def set_kk(self,kkx):
        self.kk.assign(kkx)

    def call(self, inputs):
        if(self.kk < 1e3):
            return tf.nn.sigmoid(tf.nn.leaky_relu(inputs,.5)*self.kk)+self.bias
        else:
            return tf.math.sign(inputs)/2+(.5+self.bias)