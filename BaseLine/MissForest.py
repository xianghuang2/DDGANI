import numpy as np
import pandas as pd
import sklearn.neighbors._base
import sys

from util import categorical_to_code

sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from missingpy import MissForest
import Mean
import logging
logging.basicConfig(level=logging.WARNING)
def fill_data_missForest(data_m, nan_data, con_cols, cat_cols):
    fill_mean_data = Mean.fill_data_mean(nan_data, con_cols)
    copy_ori_data = fill_mean_data.copy()
    copy_ori_data = pd.DataFrame(copy_ori_data)
    miss_code, enc = categorical_to_code(copy_ori_data, cat_cols, enc=None)
    miss_code[data_m == 0] = np.nan
    imputer = MissForest(verbose=0,criterion='squared_error', max_features=1.0,max_iter=5,n_estimators=5)
    data_imputed = imputer.fit_transform(miss_code)
    data_imputed[cat_cols] = data_imputed[cat_cols].round().astype(int)
    data_imputed = pd.DataFrame(data_imputed)
    if len(cat_cols) != 0:
        data_imputed[cat_cols] = enc.inverse_transform(data_imputed[cat_cols])
    miss_data = data_imputed.values
    return miss_data