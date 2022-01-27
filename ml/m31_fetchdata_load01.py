import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
import warnings
import os
warnings.filterwarnings(action='ignore')

x_train = np.load('fetch_covtype_datax_x_train.npy')
y_train = np.load('fetch_covtype_datax_y_train.npy')
x_test = np.load('fetch_covtype_datax_x_test.npy')
y_test = np.load('fetch_covtype_datax_y_test.npy')

report_path = 'D:\Study\_Report\\'#'D\\Study\_Report\\
file_name = os.path.abspath( __file__ ).split('\\')[3].split('.')[0]
file = open(report_path + file_name +".txt", "w")

import joblib
model = joblib.load('D:\Study\_Report\\m31_fetch_data_pickle.dat')

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
    
file.close()