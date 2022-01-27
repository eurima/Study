from concurrent.futures import thread
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.datasets import load_diabetes
from xgboost import XGBRegressor

from sklearn.feature_selection import SelectFromModel

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings(action='ignore')

x,y = load_diabetes(return_X_y=True)
# print(x.shape, y.shape) #(442, 10) (442,)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.8, shuffle = True, random_state = 66
         #, stratify= y
         )

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2 모델
model = XGBRegressor(n_jobs = -1)

#3 훈련
model.fit(x_train, y_train)

#4 평가 예측
score = model.score(x_test, y_test)
y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print("Score : ", score)
print("R2 : ", r2)
'''
노멀값 ]
Score :  0.2396187053539609
R2 :  0.2396187053539609
'''

print(np.sort(model.feature_importances_))
print('=================================')
aaa = np.sort(model.feature_importances_)
for thresh in aaa:
    selection = SelectFromModel(model, threshold=thresh, prefit= True)
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    
    print(select_x_train.shape,select_x_test.shape)
    
    selection_model = XGBRegressor(n_jobs = -1)
    #3 훈련
    selection_model.fit(select_x_train, y_train)

    #4 평가 예측
    score = selection_model.score(select_x_test, y_test)
    select_y_pred = selection_model.predict(select_x_test)

    select_r2 = r2_score(y_test, select_y_pred)
    # print("select_Score : ", score)
    # print("select_R2 : ", r2)
    
    print("Thresh = %.3f, n=%d, R2 :%2f%%" %(thresh,select_x_train.shape[1],select_r2*100))

