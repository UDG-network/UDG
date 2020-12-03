import pandas as pd
import sys 
import os
import pickle
import random
import numpy as np
import tensorflow as tf
import time
sys.path.append("..")

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from itertools import chain
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr.layers import combined_dnn_input, concat_func, add_func, FM, DNN_UDG, rig, rmse, CustomCallback,focal_loss,get_input
from deepctr.models import DeepFM, DeepFM_UDG, PNN, PNN_UDG, WDL, WDL_UDG, DIEN, DIEN_UDG, DIN, DIN_UDG
from deepctr.inputs import create_embedding_dict, embedding_lookup
from deepctr.feature_column import SparseFeat, DenseFeat, VarLenSparseFeat, get_feature_names, build_input_features, get_linear_logit,input_from_feature_columns

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[6]
epoch = 3

if sys.argv[2] == 'movielens':
    train = pd.read_csv('../dataset/ml-20m/train.csv')
    test = pd.read_csv('../dataset/ml-20m/test.csv')
    if sys.argv[3] == 'w':
        sparse_features = ["itemId", "userId", "category", 'final_gender_code', 'age_level']
        dense_features = []
        untrainable_features = []
        untrainable_features_columns = []
    else:
        sparse_features = ["itemId", "userId", "category"]
        dense_features = []
        untrainable_features = []
        untrainable_features_columns = []

elif sys.argv[2] == 'alimama':
    train = pd.read_csv('../dataset/alimama/train.csv')
    test = pd.read_csv('../dataset/alimama/test.csv')
#     train = pd.read_csv('../dataset/alimama/train_small.txt')
#     test = pd.read_csv('../dataset/alimama/test_small.txt')
    if sys.argv[3] == 'w':
        sparse_features = ["itemId", "userId", "category", 'final_gender_code', 'age_level']
        #dense_features = ['price']
        dense_features = []
        #untrainable_features = ['category', 'final_gender_code', 'age_level']
        #untrainable_features_columns = [SparseFeat(feat, train[feat].nunique(), embedding_dim=int(sys.argv[5]), 
                                                   #trainable=False) for feat in untrainable_features]
        untrainable_features = ["category", 'final_gender_code', 'age_level']
        untrainable_features_columns = [SparseFeat(feat, train[feat].nunique(), embedding_dim=int(sys.argv[5]), 
                                                   trainable=False) for feat in untrainable_features]
    else:
        sparse_features = ["itemId", "userId", "category"]
        dense_features = []
        untrainable_features = []
        untrainable_features_columns = []

elif sys.argv[2] == 'amazon':
    train = pd.read_csv('../dataset/amazon/train.csv')
    test = pd.read_csv('../dataset/amazon/test.csv')
    if sys.argv[3] == 'w':
        sparse_features = ["itemId", "userId", "category", 'final_gender_code', 'age_level']
        untrainable_features = ["category", 'final_gender_code', 'age_level']
        untrainable_features_columns = [SparseFeat(feat, train[feat].nunique(), embedding_dim=int(sys.argv[5]), 
                                                   trainable=False) for feat in untrainable_features]
    else:
        sparse_features = ["itemId", "userId", 'category']
        untrainable_features = []
        untrainable_features_columns = []
    dense_features = []


else:
    print('plz input dataset name')
    sys.exit()

udg_features = 'userId'
target = ['rating']
behavior_feature_list = ['itemId', 'category']

fixlen_feature_columns = [SparseFeat(feat, train[feat].nunique(), embedding_dim=int(sys.argv[5])) for feat in sparse_features] + [DenseFeat(feat, 1,) for feat in dense_features]
linear_feature_columns = fixlen_feature_columns 
dnn_feature_columns = fixlen_feature_columns
fixlen_feature_names = get_feature_names(fixlen_feature_columns)
train_model_input = {name: train[name] for name in fixlen_feature_names}  
test_model_input = {name: test[name] for name in fixlen_feature_names}

if sys.argv[1] in ['DIEN', 'DIEN_UDG', 'DIN', 'DIN_UDG']:
    test_model_input, test_label, max_len = get_input(test, 0, 'test')
    train_model_input, train_label, _ = get_input(train, max_len, 'train')
    fixlen_feature_columns = [SparseFeat(feat, train[feat].nunique()+1, embedding_dim=int(sys.argv[5])) for feat in sparse_features]
    fixlen_feature_columns += [DenseFeat(feat, 1,) for feat in dense_features]
    fixlen_feature_columns += [
        VarLenSparseFeat(SparseFeat('hist_itemId', train['itemId'].nunique() + 1,
                         embedding_dim = int(sys.argv[5]), embedding_name='itemId'), maxlen=max_len, 
                        length_name='seq_length'),
        VarLenSparseFeat(SparseFeat('hist_category', train['category'].nunique() + 1,
                         embedding_dim = int(sys.argv[5]), embedding_name='category'), maxlen=max_len, 
                        length_name='seq_length'),
        VarLenSparseFeat(SparseFeat('neg_hist_itemId', train['itemId'].nunique() + 1,
                         embedding_dim = int(sys.argv[5]), embedding_name='itemId'), maxlen=max_len, 
                        length_name='seq_length'),
        VarLenSparseFeat(SparseFeat('neg_hist_category', train['category'].nunique() + 1,
                         embedding_dim = int(sys.argv[5]), embedding_name='category'), maxlen=max_len, 
                        length_name='seq_length')
    ]
behavior_feature_list = ['itemId', 'category']

if sys.argv[1] == 'DeepFM_UDG':
    model = DeepFM_UDG(linear_feature_columns, dnn_feature_columns, untrainable_features_columns, 
                       (200, 80), uid_feature_name=udg_features, udg_embedding_size=int(sys.argv[5]))
elif sys.argv[1] == 'DeepFM':
    model = DeepFM(linear_feature_columns, dnn_feature_columns, [], (200, 80))
elif sys.argv[1] == 'PNN_UDG':
    model = PNN_UDG(dnn_feature_columns, untrainable_features_columns, (200, 80), uid_feature_name=udg_features, 
                    udg_embedding_size=int(sys.argv[5]))
elif sys.argv[1] == 'PNN':
    model = PNN(dnn_feature_columns, untrainable_features_columns, (200, 80))
elif sys.argv[1] == 'WDL':
    model = WDL(linear_feature_columns, dnn_feature_columns, [], (200, 80))
elif sys.argv[1] == 'WDL_UDG':
    model = WDL_UDG(linear_feature_columns, dnn_feature_columns, untrainable_features_columns, (200, 80), uid_feature_name=udg_features, udg_embedding_size=int(sys.argv[5]))
elif sys.argv[1] == 'DIEN':
    model = DIEN(fixlen_feature_columns, behavior_feature_list,
             dnn_hidden_units=[200, 80], dnn_dropout=0, gru_type="AUGRU", use_negsampling=True)
elif sys.argv[1] == 'DIEN_UDG':
    model = DIEN_UDG(fixlen_feature_columns, untrainable_features_columns, behavior_feature_list, dnn_hidden_units=[200, 80], dnn_dropout=0, gru_type="AUGRU", use_negsampling=True, uid_feature_name=udg_features, udg_embedding_size=int(sys.argv[5]))
elif sys.argv[1] == 'DIN':
    model = DIN(fixlen_feature_columns, behavior_feature_list, dnn_hidden_units=[200, 80], dnn_dropout=0)
elif sys.argv[1] == 'DIN_UDG':
    model = DIN_UDG(fixlen_feature_columns, untrainable_features_columns, behavior_feature_list, dnn_hidden_units=[200, 80], dnn_dropout=0, uid_feature_name=udg_features, udg_embedding_size=int(sys.argv[5]))
    
if sys.argv[4] == 'focal':
    model.compile("adam", loss=focal_loss, metrics=['binary_crossentropy'], )
else:
    model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )
init_lr = float(tf.keras.backend.get_value(model.optimizer.learning_rate))
lr = [init_lr, init_lr/2, init_lr/4]
history_all = {}
max_auc, min_log, min_rmse, max_rig = 0, 0, 0, 0
for x in range(epoch):
    tf.keras.backend.set_value(model.optimizer.lr, lr[x])
    history = CustomCallback()
    model.fit(train_model_input, train[target].values, batch_size=256, epochs=1, verbose=1, 
                        callbacks=[history])
    pred_ans = model.predict(test_model_input, batch_size=256)
    pred_ans = pred_ans.astype(np.float64)
    auc_value = round(roc_auc_score(test[target].values, pred_ans), 4)
    logloss_value = round(log_loss(test[target].values, pred_ans), 4)
    rmse_value = round(rmse(test[target].values, pred_ans), 4)
    rig_value = round(rig(test[target].values, pred_ans)[0], 4)
    print("test LogLoss", logloss_value) 
    print("test AUC", auc_value) 
    print("test RMSE", rmse_value)
    print("test RIG", rig_value)
#    history.result['auc'] = auc_value
#    history.result['logloss'] = logloss_value
#    history.result['rmse'] = rmse_value
#    history.result['rig'] = rig_value
#    history_all[str(x)] = history.result
#    if auc_value > max_auc:
#        max_auc = auc_value
#        min_log = logloss_value
#        min_rmse = rmse_value
#        max_rig = rig_value
#    ticks = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
#    if sys.argv[1] not in ['DIEN', 'DIEN_UDG', 'DIN', 'DIN_UDG']:
#        model.save_weights("./trained_data/model/" + sys.argv[1] + "/" + sys.argv[2]+"/" + str(x) + "_" + 
#               str(auc_value) + "_" + str(logloss_value) + "_" + str(rmse_value) + "_" + str(rig_value) + '_' +
#                       ticks + ".h5")
#if sys.argv[1] not in ['DIEN', 'DIEN_UDG', 'DIN', 'DIN_UDG']:
#    with open('./trained_data/history/' + sys.argv[1] + "/" + sys.argv[2]+"/" + 
#               str(max_auc) + "_" + str(min_log) + "_" + str(min_rmse) + "_" + str(max_rig) + '_' +
#                           ticks, 'wb') as file_pi:
#        pickle.dump(history_all, file_pi)

    
 
