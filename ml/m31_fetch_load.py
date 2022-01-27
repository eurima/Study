import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
import warnings
from sklearn.feature_selection import SelectFromModel

import os
warnings.filterwarnings(action='ignore')
    
report_path = 'D:\Study\_Report\\'#'D\\Study\_Report\\
file_name = os.path.abspath( __file__ ).split('\\')[3].split('.')[0]
file = open(report_path + file_name +".txt", "w")

x_train = np.load('fetch_covtype_datax_x_train.npy')
y_train = np.load('fetch_covtype_datax_y_train.npy')
x_test = np.load('fetch_covtype_datax_x_test.npy')
y_test = np.load('fetch_covtype_datax_y_test.npy')

model = XGBClassifier(
    tree_method = 'gpu_hist',
    predictor='gpu_predictor',
    gpu_id=0,
    n_estimators=2000,       
    learning_rate = 0.0005,    
    gamma = 1,       
    max_depth = 30,    
    min_child_weight = 5,    
    reg_lambda = 1,    
    reg_alpha = 0, 
    )
model.fit(x_train, y_train, verbose=1,
          eval_set = [(x_train, y_train),(x_test,y_test)],
          eval_metric = 'merror',
          early_stopping_rounds= 50)

score = round(model.score(x_test,y_test),4)
msg = f':Model Score : {score} \n'
print(msg)  
file.write(msg)    
y_pred = model.predict(x_test)
msg = f':Acc_score  {round(accuracy_score(y_test,y_pred),4)} \n'
print(msg)  
file.write(msg)
msg = f':f1_score : {round(f1_score(y_test,y_pred,average="macro"),4)} \n'
print(msg)  
file.write(msg)

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
file.close()

import joblib
joblib.dump(model, file_name+"02.dat")
'''

'''