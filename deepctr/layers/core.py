# -*- coding:utf-8 -*-
"""

Author:
    Weichen Shen,wcshen1994@163.com

"""

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import Zeros, glorot_normal
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.regularizers import l2

from .activation import activation_layer


class LocalActivationUnit(Layer):
    """The LocalActivationUnit used in DIN with which the representation of
    user interests varies adaptively given different candidate items.

      Input shape
        - A list of two 3D tensor with shape:  ``(batch_size, 1, embedding_size)`` and ``(batch_size, T, embedding_size)``

      Output shape
        - 3D tensor with shape: ``(batch_size, T, 1)``.

      Arguments
        - **hidden_units**:list of positive integer, the attention net layer number and units in each layer.

        - **activation**: Activation function to use in attention net.

        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix of attention net.

        - **dropout_rate**: float in [0,1). Fraction of the units to dropout in attention net.

        - **use_bn**: bool. Whether use BatchNormalization before activation or not in attention net.

        - **seed**: A Python integer to use as random seed.

      References
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
    """

    def __init__(self, hidden_units=(64, 32), activation='sigmoid', l2_reg=0, dropout_rate=0, use_bn=False, seed=1024,
                 **kwargs):
        self.hidden_units = hidden_units
        self.activation = activation
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.seed = seed
        super(LocalActivationUnit, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):

        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `LocalActivationUnit` layer should be called '
                             'on a list of 2 inputs')

        if len(input_shape[0]) != 3 or len(input_shape[1]) != 3:
            raise ValueError("Unexpected inputs dimensions %d and %d, expect to be 3 dimensions" % (
                len(input_shape[0]), len(input_shape[1])))

        if input_shape[0][-1] != input_shape[1][-1] or input_shape[0][1] != 1:
            raise ValueError('A `LocalActivationUnit` layer requires '
                             'inputs of a two inputs with shape (None,1,embedding_size) and (None,T,embedding_size)'
                             'Got different shapes: %s,%s' % (input_shape[0], input_shape[1]))
        size = 4 * \
               int(input_shape[0][-1]
                   ) if len(self.hidden_units) == 0 else self.hidden_units[-1]
        self.kernel = self.add_weight(shape=(size, 1),
                                      initializer=glorot_normal(
                                          seed=self.seed),
                                      name="kernel")
        self.bias = self.add_weight(
            shape=(1,), initializer=Zeros(), name="bias")
        self.dnn = DNN(self.hidden_units, self.activation, self.l2_reg,
                       self.dropout_rate, self.use_bn, seed=self.seed)

        self.dense = tf.keras.layers.Lambda(lambda x: tf.nn.bias_add(tf.tensordot(
            x[0], x[1], axes=(-1, 0)), x[2]))

        super(LocalActivationUnit, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, training=None, **kwargs):

        query, keys = inputs

        keys_len = keys.get_shape()[1]
        queries = K.repeat_elements(query, keys_len, 1)

        att_input = tf.concat(
            [queries, keys, queries - keys, queries * keys], axis=-1)

        att_out = self.dnn(att_input, training=training)

        attention_score = self.dense([att_out, self.kernel, self.bias])

        return attention_score

    def compute_output_shape(self, input_shape):
        return input_shape[1][:2] + (1,)

    def compute_mask(self, inputs, mask):
        return mask

    def get_config(self, ):
        config = {'activation': self.activation, 'hidden_units': self.hidden_units,
                  'l2_reg': self.l2_reg, 'dropout_rate': self.dropout_rate, 'use_bn': self.use_bn, 'seed': self.seed}
        base_config = super(LocalActivationUnit, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DNN(Layer):
    """The Multi Layer Percetron

      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.

      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input with shape ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.

      Arguments
        - **hidden_units**:list of positive integer, the layer number and units in each layer.

        - **activation**: Activation function to use.

        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.

        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.

        - **use_bn**: bool. Whether use BatchNormalization before activation or not.

        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False, seed=1024, **kwargs):
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        super(DNN, self).__init__(**kwargs)

    def build(self, input_shape):
        # if len(self.hidden_units) == 0:
        #     raise ValueError("hidden_units is empty")
        input_size = input_shape[-1]
        hidden_units = [int(input_size)] + list(self.hidden_units)
        self.kernels = [self.add_weight(name='kernel' + str(i),
                                        shape=(
                                            hidden_units[i], hidden_units[i + 1]),
                                        initializer=glorot_normal(
                                            seed=self.seed),
                                        regularizer=l2(self.l2_reg),
                                        trainable=True) for i in range(len(self.hidden_units))]
        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(self.hidden_units[i],),
                                     initializer=Zeros(),
                                     trainable=True) for i in range(len(self.hidden_units))]
        if self.use_bn:
            self.bn_layers = [tf.keras.layers.BatchNormalization() for _ in range(len(self.hidden_units))]

        self.dropout_layers = [tf.keras.layers.Dropout(self.dropout_rate, seed=self.seed + i) for i in
                               range(len(self.hidden_units))]

        self.activation_layers = [activation_layer(self.activation) for _ in range(len(self.hidden_units))]

        super(DNN, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, training=None, **kwargs):

        deep_input = inputs

        for i in range(len(self.hidden_units)):
            fc = tf.nn.bias_add(tf.tensordot(
                deep_input, self.kernels[i], axes=(-1, 0)), self.bias[i])
            # fc = Dense(self.hidden_size[i], activation=None, \
            #           kernel_initializer=glorot_normal(seed=self.seed), \
            #           kernel_regularizer=l2(self.l2_reg))(deep_input)
            if self.use_bn:
                fc = self.bn_layers[i](fc, training=training)

            fc = self.activation_layers[i](fc)

            fc = self.dropout_layers[i](fc, training=training)
            deep_input = fc

        return deep_input

    def compute_output_shape(self, input_shape):
        if len(self.hidden_units) > 0:
            shape = input_shape[:-1] + (self.hidden_units[-1],)
        else:
            shape = input_shape

        return tuple(shape)

    def get_config(self, ):
        config = {'activation': self.activation, 'hidden_units': self.hidden_units,
                  'l2_reg': self.l2_reg, 'use_bn': self.use_bn, 'dropout_rate': self.dropout_rate, 'seed': self.seed}
        base_config = super(DNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# class UDG(Layer):
 
#     def __init__(self, hidden_units, activation='sigmoid', l2_reg=0, dropout_rate=0, use_bn=False, seed=1024, **kwargs):
#         self.hidden_units = hidden_units
#         self.activation = activation
#         self.dropout_rate = dropout_rate
#         self.seed = seed
#         self.l2_reg = l2_reg
#         self.use_bn = use_bn
#         super(UDG, self).__init__(**kwargs)

#     def build(self, input_shape):
#         # if len(self.hidden_units) == 0:
#         #     raise ValueError("hidden_units is empty")
#         input_size = input_shape[-1]
#         hidden_units = [int(input_size)] + list(self.hidden_units)
#         self.kernels = [self.add_weight(name='udg_weight' + str(i),
#                                         shape=(hidden_units[i], hidden_units[i + 1]),
#                                         initializer=glorot_normal(seed=self.seed),
#                                         regularizer=l2(self.l2_reg),
#                                         trainable=True) for i in range(len(self.hidden_units))]
#         self.bias = [self.add_weight(name='udg_bias' + str(i),
#                                      shape=(self.hidden_units[i],),
#                                      initializer=Zeros(),
#                                      trainable=True) for i in range(len(self.hidden_units))]
#         if self.use_bn:
#             self.bn_layers = [tf.keras.layers.BatchNormalization() for _ in range(len(self.hidden_units))]

#         self.dropout_layers = [tf.keras.layers.Dropout(self.dropout_rate, seed=self.seed + i) for i in
#                                range(len(self.hidden_units))]
        
#         self.activation_layers = []
#         for i, x in enumerate(hidden_units):
#             if i != len(hidden_units)-1:
#                 self.activation_layers.append(activation_layer('relu'))
#             else:
#                 self.activation_layers.append(activation_layer('sigmoid'))

#         super(UDG, self).build(input_shape)  # Be sure to call this somewhere!

#     def call(self, inputs, training=None, **kwargs):

#         deep_input = inputs

#         for i in range(len(self.hidden_units)):
#             fc = tf.nn.bias_add(tf.tensordot(
#                 deep_input, self.kernels[i], axes=(-1, 0)), self.bias[i])
#             if self.use_bn:
#                 fc = self.bn_layers[i](fc, training=training)

#             fc = self.activation_layers[i](fc)

#             fc = self.dropout_layers[i](fc, training=training)
#             deep_input = fc

#         return deep_input

#     def compute_output_shape(self, input_shape):
#         if len(self.hidden_units) > 0:
#             shape = input_shape[:-1] + (self.hidden_units[-1],)
#         else:
#             shape = input_shape

#         return tuple(shape)

#     def get_config(self, ):
#         config = {'activation': self.activation, 'hidden_units': self.hidden_units,
#                   'l2_reg': self.l2_reg, 'use_bn': self.use_bn, 'dropout_rate': self.dropout_rate, 'seed': self.seed}
#         base_config = super(UDG, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

class DNN_UDG(Layer):
 
    def __init__(self, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False, seed=1024,
                 udg_embedding_size=128, udg_embedding_layer=3, **kwargs):
        self.hidden_units = hidden_units
        self.udg_embedding_size = int(udg_embedding_size)
        self.udg_embedding_layer = int(udg_embedding_layer)
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        super(DNN_UDG, self).__init__(**kwargs)

    def build(self, input_shape):
        # if len(self.hidden_units) == 0:
        #     raise ValueError("hidden_units is empty")
        input_size = input_shape[-1]
        print(input_size, self.udg_embedding_size)
        hidden_units = [int(input_size)-self.udg_embedding_size] + list(self.hidden_units) 
        udg_units = []
        for i, x in enumerate(hidden_units):
            tmp = []
            for j in range(self.udg_embedding_layer):
                if j == 0:
                    tmp.append(self.udg_embedding_size)
                else:
                    tmp.append(x)
            udg_units.append(tmp)
        print(udg_units)
        #udg_units = [[128,384,384],[128,200,200],[128,80,80]] 
        self.udg_kernels = [self.add_weight(name='udg_kernel' + str(i) + '_' + str(j),
                                        shape=(udg_units[i][j], hidden_units[i]),
                                        initializer=glorot_normal(seed=self.seed),
                                        regularizer=l2(self.l2_reg),
                                        trainable=True) for i in range(len(udg_units)) for j in range(3)]
        self.udg_bias = [self.add_weight(name='udg_bias' + str(i) + '_' + str(j),
                                        shape=(hidden_units[i],),
                                        initializer=glorot_normal(seed=self.seed),
                                        regularizer=l2(self.l2_reg),
                                        trainable=True) for i in range(len(udg_units)) for j in range(3)]
        self.kernels = [self.add_weight(name='kernel' + str(i),
                                        shape=(hidden_units[i], hidden_units[i + 1]),
                                        initializer=glorot_normal(seed=self.seed),
                                        regularizer=l2(self.l2_reg),
                                        trainable=True) for i in range(len(self.hidden_units))]
        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(self.hidden_units[i],),
                                     initializer=Zeros(),
                                     trainable=True) for i in range(len(self.hidden_units))]
        if self.use_bn:
            self.bn_layers = [tf.keras.layers.BatchNormalization() for _ in range(len(self.hidden_units))]

        self.dropout_layers = [tf.keras.layers.Dropout(self.dropout_rate, seed=self.seed + i) for i in
                               range(len(self.hidden_units))]
        
        self.udg_activation_layers = []
        for i, x in enumerate(hidden_units):
            for j in range(self.udg_embedding_layer):
                if j != self.udg_embedding_layer-1:
                    self.udg_activation_layers.append(activation_layer('relu'))
                else:
                    self.udg_activation_layers.append(activation_layer('sigmoid'))
                
        self.activation_layers = [activation_layer(self.activation) for _ in range(len(self.hidden_units))]

        super(DNN_UDG, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, training=None, **kwargs):
        input_size = int(inputs.shape[-1])
        deep_input = inputs[:,0:input_size-self.udg_embedding_size]
        udg_input = inputs[:,input_size-self.udg_embedding_size:]
        for i in range(len(self.hidden_units)):
            udg_input_tmp = udg_input
            for j in range(self.udg_embedding_layer):
                udg_tmp = tf.nn.bias_add(tf.tensordot(udg_input_tmp, self.udg_kernels[i*3+j], 
                                                      axes=(-1, 0)), self.udg_bias[i*3+j])
                udg_tmp = self.udg_activation_layers[i*3+j](udg_tmp)
                udg_input_tmp = udg_tmp
            fc_udg = tf.multiply(deep_input, udg_input_tmp)
            fc = tf.nn.bias_add(tf.tensordot(fc_udg, self.kernels[i], axes=(-1, 0)), self.bias[i])
            if self.use_bn:
                fc = self.bn_layers[i](fc, training=training)
            fc = self.activation_layers[i](fc)
            fc = self.dropout_layers[i](fc, training=training)
            
            if i+1 == len(self.hidden_units):
                for j in range(self.udg_embedding_layer):
                    udg_tmp = tf.nn.bias_add(tf.tensordot(udg_input, self.udg_kernels[(i+1)*3+j], 
                                                          axes=(-1, 0)), self.udg_bias[(i+1)*3+j])
                    udg_tmp = self.udg_activation_layers[(i+1)*3+j](udg_tmp)
                    udg_input = udg_tmp
                fc = tf.multiply(fc, udg_tmp)
                
            deep_input = fc

        return deep_input

    def compute_output_shape(self, input_shape):
        if len(self.hidden_units) > 0:
            shape = input_shape[:-1]-self.udg_embedding_size + (self.hidden_units[-1],)
        else:
            shape = input_shape-self.udg_embedding_size

        return tuple(shape)

    def get_config(self, ):
        config = {'activation': self.activation, 
                  'udg_embedding_size': self.udg_embedding_size,
                  'udg_embedding_layer': self.udg_embedding_layer,
                  'hidden_units': self.hidden_units,
                  'l2_reg': self.l2_reg, 
                  'use_bn': self.use_bn, 
                  'dropout_rate': self.dropout_rate, 
                  'seed': self.seed}
        base_config = super(DNN_UDG, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PredictionLayer(Layer):
    """
      Arguments
         - **task**: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss

         - **use_bias**: bool.Whether add bias term or not.
    """

    def __init__(self, task='binary', use_bias=True, **kwargs):
        if task not in ["binary", "multiclass", "regression"]:
            raise ValueError("task must be binary,multiclass or regression")
        self.task = task
        self.use_bias = use_bias
        super(PredictionLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        if self.use_bias:
            self.global_bias = self.add_weight(
                shape=(1,), initializer=Zeros(), name="global_bias")

        # Be sure to call this somewhere!
        super(PredictionLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs
        if self.use_bias:
            x = tf.nn.bias_add(x, self.global_bias, data_format='NHWC')
        if self.task == "binary":
            x = tf.sigmoid(x)

        output = tf.reshape(x, (-1, 1))

        return output

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self, ):
        config = {'task': self.task, 'use_bias': self.use_bias}
        base_config = super(PredictionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


