# from sklearn.datasets import fetch_covtype
# datasets = fetch_covtype()

import pickle
# save
# with open('fetch_covtype_data.pickle', 'wb') as f:
#     pickle.dump(datasets, f, pickle.HIGHEST_PROTOCOL)
    
# load
with open('fetch_covtype_data.pickle', 'rb') as f:
    datasets = pickle.load(f)
    
x = datasets.data
y = datasets.target


import pandas as pd
import time
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=66)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
#훈련
histt = model = XGBClassifier(
    n_estimators=100000,       
    learning_rate = 0.01,    
    gamma = 1,       
    max_depth = 20,    
    min_child_weight = 5,    
    reg_lambda = 1,    
    reg_alpha = 0, 
    n_jobs=-1 
)
start = time.time()
model.fit(x_train, y_train, verbose=1,
          eval_set = [(x_train, y_train),(x_test,y_test)],
          eval_metric = 'mlogloss',
          early_stopping_rounds= 50
          )
end = time.time()-start
print("걸린 시간 : ", round(end, 2))
score = round(model.score(x_test, y_test),5)
print("score : ", score)
