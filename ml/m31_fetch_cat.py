import os
import sys
import platform
import random
import math
from typing import List ,Dict, Tuple

import pandas as pd
import numpy as np
 
import sklearn 
from sklearn.model_selection import StratifiedKFold , KFold
from sklearn.metrics import f1_score 
from sklearn.preprocessing import StandardScaler

from catboost import Pool,CatBoostClassifier

x_train = np.load('fetch_covtype_datax_x_train.npy')
y_train = np.load('fetch_covtype_datax_y_train.npy')
x_test = np.load('fetch_covtype_datax_x_test.npy')
y_test = np.load('fetch_covtype_datax_y_test.npy')

cat_features = x_train.columns[x_train.nunique() > 2].tolist()