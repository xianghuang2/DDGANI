import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

import Mean
from util import categorical_to_code


def sim_impute(miss_data, con_cols, cat_cols, data_m):
    fill_mean_data = Mean.fill_data_mean(miss_data, con_cols)
    copy_ori_data = fill_mean_data.copy()
    copy_ori_data = pd.DataFrame(copy_ori_data)
    miss_code, enc = categorical_to_code(copy_ori_data, cat_cols, enc=None)
    miss_code[data_m == 0] = np.nan
    hot_deck_imputer = KNNImputer(n_neighbors=2, weights="uniform")
    data_imputed = hot_deck_imputer.fit_transform(miss_code)
    data_imputed[cat_cols] = data_imputed[cat_cols].round().astype(int)
    data_imputed = pd.DataFrame(data_imputed)
    if len(cat_cols) != 0:
        data_imputed[cat_cols] = enc.inverse_transform(data_imputed[cat_cols])
    miss_data = data_imputed.values
    return miss_data
