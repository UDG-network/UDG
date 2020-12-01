# -*- coding:utf-8 -*-
"""

Author:
    Weichen Shen,wcshen1994@163.com

"""
import tensorflow as tf
from tensorflow.python.keras import backend as K
import numpy as np
from tensorflow.python import keras
from sklearn.metrics import mean_squared_error
from tensorflow.python.keras.layers import Flatten
from tqdm import tqdm
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

class CustomCallback(keras.callbacks.Callback):
    
    def on_train_begin(self, logs=None):
        self.result = {'batch_id':[], 'batch_loss':[], 'batch_binary_crossentropy':[], 
                      'auc':0, 'logloss':0, 'rmse':0, 'rig':0}

    def on_train_batch_end(self, batch, logs=None):
        if batch % 5 == 0:
            self.result['batch_id'].append(batch)
            self.result['batch_loss'].append(logs['loss'])
            self.result['batch_binary_crossentropy'].append(logs['binary_crossentropy'])

class NoMask(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(NoMask, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(NoMask, self).build(input_shape)

    def call(self, x, mask=None, **kwargs):
        return x

    def compute_mask(self, inputs, mask):
        return None


class Hash(tf.keras.layers.Layer):
    """
    hash the input to [0,num_buckets)
    if mask_zero = True,0 or 0.0 will be set to 0,other value will be set in range[1,num_buckets)
    """

    def __init__(self, num_buckets, mask_zero=False, **kwargs):
        self.num_buckets = num_buckets
        self.mask_zero = mask_zero
        super(Hash, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(Hash, self).build(input_shape)

    def call(self, x, mask=None, **kwargs):
        if x.dtype != tf.string:
            x = tf.as_string(x, )
        try:
            hash_x = tf.string_to_hash_bucket_fast(x, self.num_buckets if not self.mask_zero else self.num_buckets - 1,
                                                    name=None)  # weak hash
        except:
            hash_x = tf.strings.to_hash_bucket_fast(x, self.num_buckets if not self.mask_zero else self.num_buckets - 1,
                                               name=None)  # weak hash
        if self.mask_zero:
            mask_1 = tf.cast(tf.not_equal(x, "0"), 'int64')
            mask_2 = tf.cast(tf.not_equal(x, "0.0"), 'int64')
            mask = mask_1 * mask_2
            hash_x = (hash_x + 1) * mask
        return hash_x

    def compute_mask(self, inputs, mask):
        return None

    def get_config(self, ):
        config = {'num_buckets': self.num_buckets, 'mask_zero': self.mask_zero}
        base_config = super(Hash, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Linear(tf.keras.layers.Layer):

    def __init__(self, l2_reg=0.0, mode=0, use_bias=False, **kwargs):

        self.l2_reg = l2_reg
        # self.l2_reg = tf.contrib.layers.l2_regularizer(float(l2_reg_linear))
        if mode not in [0, 1, 2]:
            raise ValueError("mode must be 0,1 or 2")
        self.mode = mode
        self.use_bias = use_bias
        super(Linear, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.use_bias:
            self.bias = self.add_weight(name='linear_bias',
                                        shape=(1,),
                                        initializer=tf.keras.initializers.Zeros(),
                                        trainable=True)
        if self.mode == 1:
            self.kernel = self.add_weight(
                'linear_kernel',
                shape=[int(input_shape[-1]), 1],
                initializer=tf.keras.initializers.glorot_normal(),
                regularizer=tf.keras.regularizers.l2(self.l2_reg),
                trainable=True)
        elif self.mode == 2 :
            self.kernel = self.add_weight(
                'linear_kernel',
                shape=[int(input_shape[1][-1]), 1],
                initializer=tf.keras.initializers.glorot_normal(),
                regularizer=tf.keras.regularizers.l2(self.l2_reg),
                trainable=True)

        super(Linear, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):
        if self.mode == 0:
            sparse_input = inputs
            linear_logit = reduce_sum(sparse_input, axis=-1, keep_dims=True)
        elif self.mode == 1:
            dense_input = inputs
            fc = tf.tensordot(dense_input, self.kernel, axes=(-1, 0))
            linear_logit = fc
        else:
            sparse_input, dense_input = inputs
            fc = tf.tensordot(dense_input, self.kernel, axes=(-1, 0))
            linear_logit = reduce_sum(sparse_input, axis=-1, keep_dims=False) + fc
        if self.use_bias:
            linear_logit += self.bias

        return linear_logit

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def compute_mask(self, inputs, mask):
        return None

    def get_config(self, ):
        config = {'mode': self.mode, 'l2_reg': self.l2_reg,'use_bias':self.use_bias}
        base_config = super(Linear, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def concat_func(inputs, axis=-1, mask=False):
    if not mask:
        inputs = list(map(NoMask(), inputs))
    if len(inputs) == 1:
        return inputs[0]
    else:
        return tf.keras.layers.Concatenate(axis=axis)(inputs)


def reduce_mean(input_tensor,
               axis=None,
               keep_dims=False,
               name=None,
               reduction_indices=None):
    if tf.__version__ < '2.0.0':
        return tf.reduce_mean(input_tensor,
                   axis=axis,
                   keep_dims=keep_dims,
                   name=name,
                   reduction_indices=reduction_indices)
    else:
        return  tf.reduce_mean(input_tensor,
                   axis=axis,
                   keepdims=keep_dims,
                   name=name)


def reduce_sum(input_tensor,
               axis=None,
               keep_dims=False,
               name=None,
               reduction_indices=None):
    if tf.__version__ < '2.0.0':
        return tf.reduce_sum(input_tensor,
                   axis=axis,
                   keep_dims=keep_dims,
                   name=name,
                   reduction_indices=reduction_indices)
    else:
        return  tf.reduce_sum(input_tensor,
                   axis=axis,
                   keepdims=keep_dims,
                   name=name)

def reduce_max(input_tensor,
               axis=None,
               keep_dims=False,
               name=None,
               reduction_indices=None):
    if tf.__version__ < '2.0.0':
        return tf.reduce_max(input_tensor,
                   axis=axis,
                   keep_dims=keep_dims,
                   name=name,
                   reduction_indices=reduction_indices)
    else:
        return  tf.reduce_max(input_tensor,
                   axis=axis,
                   keepdims=keep_dims,
                   name=name)

def div(x, y, name=None):
    if tf.__version__ < '2.0.0':
        return tf.div(x, y, name=name)
    else:
        return tf.divide(x, y, name=name)

def softmax(logits, dim=-1, name=None):
    if tf.__version__ < '2.0.0':
        return tf.nn.softmax(logits, dim=dim, name=name)
    else:
        return tf.nn.softmax(logits, axis=dim, name=name)

def get_input(train, max_len, signal):
    input_dict = {'userId': [], 'final_gender_code': [], 'itemId': [], 
                'category': [], 'age_level':[],
                'hist_itemId': [], 'hist_category': [],
                'neg_hist_itemId': [], 'neg_hist_category': [],
                "seq_length": []}
    train_hist_item_list = []
    train_hist_cate_list = []
    train_neg_hist_item_list = []
    train_neg_hist_cate_list = []
    input_dict['userId'] = train['userId'].values
    input_dict['final_gender_code'] = train['final_gender_code'].values
    input_dict['category'] = train['category'].values
    input_dict['age_level'] = train['age_level'].values
    input_dict['itemId'] = train['itemId'].values
    train_label = train['rating'].values
    for x in tqdm(range(len(train))):       
        train_hist_item_list.append(list(map(int, eval(train['hist_item_list'][x]))))
        train_hist_cate_list.append(list(map(int, eval(train['hist_cate_list'][x]))))
        train_neg_hist_item_list.append(list(map(int, eval(train['neg_hist_item_list'][x]))))
        train_neg_hist_cate_list.append(list(map(int, eval(train['neg_hist_item_list'][x]))))
    all_len = list(map(len, train_hist_item_list))
    if signal == 'test':
        train_hist_item_list = pad_sequences(train_hist_item_list, maxlen=max(all_len), padding='post', )
        train_hist_cate_list = pad_sequences(train_hist_cate_list, maxlen=max(all_len), padding='post', )
        train_neg_hist_item_list = pad_sequences(train_neg_hist_item_list, maxlen=max(all_len), padding='post', )
        train_neg_hist_cate_list = pad_sequences(train_neg_hist_cate_list, maxlen=max(all_len), padding='post', )
    else:
        train_hist_item_list = pad_sequences(train_hist_item_list, maxlen=max_len, padding='post', )
        train_hist_cate_list = pad_sequences(train_hist_cate_list, maxlen=max_len, padding='post', )
        train_neg_hist_item_list = pad_sequences(train_neg_hist_item_list, maxlen=max_len, padding='post', )
        train_neg_hist_cate_list = pad_sequences(train_neg_hist_cate_list, maxlen=max_len, padding='post', )
    input_dict['hist_itemId'] = train_hist_item_list
    input_dict['hist_category'] = train_hist_cate_list
    input_dict['neg_hist_itemId'] = train_neg_hist_item_list
    input_dict['neg_hist_category'] = train_neg_hist_cate_list
    input_dict['seq_length'] = np.array(all_len)

    return input_dict, train_label, max(all_len)

def focal_loss(y_true, y_pred, gamma=2., alpha=0.25):
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def rig(y_true, y_pred):
    p_hat = np.average(y_true)
    h = - p_hat * np.log(p_hat) - (1-p_hat) * np.log(1-p_hat)
    ce = 0.
    for i in range(len(y_true)):
        if y_pred[i] < 0.00001:
            y_pred[i] = 0.00001
        elif y_pred[i] > 1 - 0.00001:
            y_pred[i] = 1 - 0.00001
        ce += y_true[i] * np.log(y_pred[i]) + (1-y_true[i]) * np.log(1-y_pred[i])
    ce = ce / len(y_true)
    rig = (h + ce) / h
    return rig

class Add(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Add, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(Add, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if not isinstance(inputs,list):
            return inputs
        if len(inputs) == 1  :
            return inputs[0]
        if len(inputs) == 0:
            return tf.constant([[0.0]])

        return tf.keras.layers.add(inputs)
    
def add_func(inputs):
    return Add()(inputs)


def combined_dnn_input(sparse_embedding_list, dense_value_list, udg_label=0, udg_embedding_list=[]):
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_dnn_input = Flatten()(concat_func(sparse_embedding_list))
        dense_dnn_input = Flatten()(concat_func(dense_value_list))
        if udg_label:
            udg_dnn_input = Flatten()(concat_func(udg_embedding_list))
            return concat_func([sparse_dnn_input, dense_dnn_input, udg_dnn_input])
        else:
            return concat_func([sparse_dnn_input, dense_dnn_input])
    elif len(sparse_embedding_list) > 0:
        if udg_label:
            sparse_dnn_input = Flatten()(concat_func(sparse_embedding_list))
            udg_dnn_input = Flatten()(concat_func(udg_embedding_list))
            print(sparse_dnn_input, udg_dnn_input)
            return concat_func([sparse_dnn_input, udg_dnn_input])
        else:
            return Flatten()(concat_func(sparse_embedding_list))
    elif len(dense_value_list) > 0:
        if udg_label:
            udg_dnn_input = Flatten()(concat_func(udg_embedding_list))
            dense_dnn_input = Flatten()(concat_func(dense_value_list))
            return concat_func([dense_dnn_input, udg_dnn_input])
        else:
            return Flatten()(concat_func(dense_value_list))
    else:
        raise NotImplementedError("dnn_feature_columns can not be empty list")