import pandas as pd
import sys 
import os
import pickle
import random
import numpy as np
import tensorflow as tf
sys.path.append("..")

import matplotlib.pyplot as plt
from itertools import chain
from sklearn.metrics import mean_squared_error
from keras.callbacks import Callback
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr.layers import combined_dnn_input, concat_func, add_func, FM, DNN_UDG
from deepctr.models import DeepFM, DeepFM_UDG
from deepctr.inputs import create_embedding_dict, embedding_lookup
from deepctr.feature_column import SparseFeat, DenseFeat, VarLenSparseFeat, get_feature_names, build_input_features, get_linear_logit,input_from_feature_columns

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if sys.argv[1] == 'small':
    train = pd.read_csv('../dataset/ml-20m/train_small.txt')
    test = pd.read_csv('../dataset/ml-20m/test_small.txt')
elif sys.argv[1] == 'big':
    train = pd.read_csv('../dataset/ml-20m/train.txt')
    test = pd.read_csv('../dataset/ml-20m/test.txt')
else:
    print('plz input big or small')
    sys.exit()
    
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
def rig(y_true, y_pred):
    p_hat = np.average(y_true)
    h = - p_hat * np.log(p_hat) - (1-p_hat) * np.log(1-p_hat)
    ce = 0.
    for i in range(len(y_true)):
        if y_pred[i] == 0:
            y_pred[i] = 0.00000001
        elif y_pred[i] == 1:
            y_pred[i] = 1 - 0.00000001
        ce += y_true[i] * np.log(y_pred[i]) + (1-y_true[i]) * np.log(1-y_pred[i])
    ce = ce / len(y_true)
    rig = (h + ce) / h
    return rig
    
sparse_features = ["movieId", "userId", "genres"]
udg_features = 'userId'
target = ['rating']

fixlen_feature_columns = [SparseFeat(feat, train[feat].nunique(), embedding_dim=128) for feat in sparse_features]
linear_feature_columns = fixlen_feature_columns
dnn_feature_columns = fixlen_feature_columns

feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

train_model_input = {name: train[name] for name in sparse_features}  #
test_model_input = {name: test[name] for name in sparse_features}  #

#history = LossHistory()
model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary', 
                       dnn_hidden_units=(200, 80))
model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )

for x in range(5):
    history = model.fit(train_model_input, train[target].values, batch_size=256, epochs=1, verbose=1)
    pred_ans = model.predict(test_model_input, batch_size=256)
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4)) 
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4)) 
    print("test RMSE", round(rmse(test[target].values, pred_ans), 4))
    print("test RIG", round(rig(test[target].values, pred_ans)[0], 4))
    
    model.save("./trained_data/normal_model_" + str(x))
    with open('./trained_data/normal_history_' + str(x), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)