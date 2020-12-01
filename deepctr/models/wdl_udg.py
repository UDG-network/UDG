# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,wcshen1994@163.com

Reference:
    [1] Cheng H T, Koc L, Harmsen J, et al. Wide & deep learning for recommender systems[C]//Proceedings of the 1st Workshop on Deep Learning for Recommender Systems. ACM, 2016: 7-10.(https://arxiv.org/pdf/1606.07792.pdf)
"""

from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Model

from collections import OrderedDict
from ..feature_column import build_input_features, get_linear_logit, input_from_feature_columns
from ..layers.core import PredictionLayer, DNN_UDG
from ..layers.utils import add_func, combined_dnn_input
from ..layers.interaction import FM
from ..inputs import create_embedding_dict, embedding_lookup


def WDL_UDG(linear_feature_columns, dnn_feature_columns, untrain_feature_columns, 
            dnn_hidden_units=(128, 128), l2_reg_linear=1e-5,
        l2_reg_embedding=1e-5, l2_reg_dnn=0, seed=1024, dnn_dropout=0, dnn_activation='relu',
        task='binary', uid_feature_name='', udg_embedding_size=128):
    """Instantiates the Wide&Deep Learning architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to wide part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """

    features = build_input_features(
        linear_feature_columns + dnn_feature_columns)
    
    uid_features = OrderedDict()
    uid_features[uid_feature_name] = features[uid_feature_name]
    uid_feature_columns = [x for x in linear_feature_columns if x.name == uid_feature_name]

    inputs_list = list(features.values())

    linear_logit = get_linear_logit(features, linear_feature_columns, untrain_feature_columns, 
                                    seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear)

    sparse_embedding_list, dense_value_list, untrain_embedding_list = input_from_feature_columns(features, dnn_feature_columns, 
                                                                                                 untrain_feature_columns,
                                                                                                 l2_reg_embedding, seed)
    
    uid_embedding_dict = create_embedding_dict(uid_feature_columns, [], 0.00001, prefix='udg_', 
                                           seq_mask_zero=True)
    uid_emb_list = embedding_lookup(uid_embedding_dict, uid_features, uid_feature_columns, [],
                                    return_feat_list=[], to_list=True)
    uid_emb_list = uid_emb_list + untrain_embedding_list
    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list, udg_label=1, udg_embedding_list=uid_emb_list)
    print(dnn_input)
    udg_embedding_size = (len(untrain_feature_columns)+1)*udg_embedding_size
    dnn_out = DNN_UDG(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, False, seed, 
                      udg_embedding_size=udg_embedding_size)(dnn_input)
    dnn_logit = Dense(1, use_bias=False, activation=None)(dnn_out)
    final_logit = add_func([dnn_logit, linear_logit])
    output = PredictionLayer(task)(final_logit)

    model = Model(inputs=inputs_list, outputs=output)
    return model
