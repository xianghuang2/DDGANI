"""
Use the not-MIWAE on UCI data
"""
import numpy as np
import pandas as pd
import os
import sys

import tensorflow as tf
import torch

import Mean
from utils.util import categorical_to_code, Data_convert, get_M_by_data_m, reconvert_data

sys.path.append(os.getcwd())
from BaseLine.not_MIWAE.notMIWAE import notMIWAE
from BaseLine.not_MIWAE import trainer
from BaseLine.not_MIWAE import utils
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor


def get_notMIWAE_filled(data_m, nan_data, continuous_cols, categorical_cols, value_cat, enc, values, device):
    # ---- data settings
    n_hidden = 128
    n_samples = 20
    max_iter = 1000
    batch_size = 16
    L = 1000
    tf.random.set_seed(42)

    # Mean impute
    fill_mean_data = Mean.fill_data_mean(nan_data, continuous_cols)
    # ---- choose the missing model
    mprocess = 'selfmasking_known'
    name = '/tmp/uci/task01/best'
    N, D = fill_mean_data.shape
    dl = D - 1

    # ---- standardize data
    attention_impute_data = pd.DataFrame(fill_mean_data)
    attention_impute_data.columns = values
    cat_to_code_data, enc = categorical_to_code(attention_impute_data.copy(), value_cat, enc)
    cat_to_code_data.columns = [x for x in range(cat_to_code_data.shape[1])]
    fields, feed_data = Data_convert(cat_to_code_data, "minmax", continuous_cols)

    M_tensor = get_M_by_data_m(data_m, fields, device)
    M_numpy = M_tensor.cpu().numpy()

    X = feed_data.copy().values
    X[M_numpy==0] = np.nan
    Xnan = X.copy()
    Xval = X.copy()
    Xz = feed_data.values * M_numpy
    S = M_numpy

    # ---------------------- #
    # ---- fit not-MIWAE---- #
    # ---------------------- #
    notmiwae = notMIWAE(Xnan, Xval, n_latent=dl, n_samples=n_samples, n_hidden=n_hidden, missing_process=mprocess, name=name)

    # ---- do the training
    trainer.train(notmiwae, batch_size=batch_size, max_iter=max_iter, name=name + 'notmiwae')

    # ---- find imputation RMSE
    filled_data = utils.not_imputation(notmiwae, Xz, Xnan, S, L)

    #  reconvert data
    code = torch.tensor(filled_data)
    impute_data = reconvert_data(code, fields, value_cat, values, attention_impute_data.copy(), data_m, enc)
    return impute_data.values
