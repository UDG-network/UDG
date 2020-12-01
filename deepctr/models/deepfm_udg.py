# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,wcshen1994@163.com

Reference:
    [1] Guo H, Tang R, Ye Y, et al. Deepfm: a factorization-machine based neural network for ctr prediction[J]. arXiv preprint arXiv:1703.04247, 2017.(https://arxiv.org/abs/1703.04247)

"""

from itertools import chain

import tensorflow as tf

from collections import OrderedDict
from ..feature_column import build_input_features, get_linear_logit, DEFAULT_GROUP_NAME, input_from_feature_columns, SparseFeat
from ..layers.core import PredictionLayer, DNN_UDG
from ..layers.interaction import FM
from ..inputs import create_embedding_dict, embedding_lookup
from ..layers.utils import concat_func, add_func, combined_dnn_input


def DeepFM_UDG(linear_feature_columns, dnn_feature_columns, untrain_feature_columns, 
               fm_group=[DEFAULT_GROUP_NAME], udg_hidden_units=(128, 128),
               dnn_hidden_units=(200, 80), l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, seed=1024,
               dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False, task='binary', 
               uid_feature_name='', udg_embedding_size=128):
    """Instantiates the DeepFM Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param fm_group: list, group_name of features that will be used to do feature interactions.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """

    features = build_input_features(linear_feature_columns + dnn_feature_columns)
    uid_features = OrderedDict()
    uid_features[uid_feature_name] = features[uid_feature_name]
    uid_feature_columns = [x for x in linear_feature_columns if x.name == uid_feature_name]

    inputs_list = list(features.values())

    linear_logit = get_linear_logit(features, linear_feature_columns, untrain_feature_columns, seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear)

    group_embedding_dict, dense_value_list, untrain_embedding_dict = input_from_feature_columns(features, 
                                                                                               dnn_feature_columns, 
                                                                                               untrain_feature_columns, 
                                                                                                l2_reg_embedding,
                                                                                                seed, support_group=True)

    fm_logit = add_func([FM()(concat_func(v, axis=1))
                         for k, v in group_embedding_dict.items() if k in fm_group])
    
    uid_embedding_dict = create_embedding_dict(uid_feature_columns, [], 0.00001, prefix='udg_', 
                                           seq_mask_zero=True)
    uid_emb_list = embedding_lookup(uid_embedding_dict, uid_features, uid_feature_columns, [],
                                    return_feat_list=[], to_list=True)
    uid_emb_list = uid_emb_list + untrain_embedding_dict

    dnn_input = combined_dnn_input(list(chain.from_iterable(
        group_embedding_dict.values())), dense_value_list, udg_label=1, udg_embedding_list=uid_emb_list)
    
    udg_embedding_size = (len(untrain_feature_columns)+1)*udg_embedding_size
    #udg_input = combined_dnn_input(uid_emb_list, [])
    print(dnn_input.shape)
    #udg_output = []
#     for x in [dnn_input.shape[1].value] + list(dnn_hidden_units):
#         udg_hidden_units = (x, x, x)
#         output = UDG(udg_hidden_units, udg_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed)(udg_input)
#         udg_output.append(output)
    dnn_output = DNN_UDG(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                     dnn_use_bn, seed, udg_embedding_size=udg_embedding_size)(dnn_input)
    
    dnn_logit = tf.keras.layers.Dense(
        1, use_bias=False, activation=None)(dnn_output)

    final_logit = add_func([linear_logit, fm_logit, dnn_logit])

    output = PredictionLayer(task)(final_logit)
    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model
