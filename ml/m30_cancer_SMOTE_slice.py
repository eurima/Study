import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import warnings
import os
warnings.filterwarnings(action='ignore')
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target 
index_list = np.where(y==0)
index_list = index_list[0][111:]
new_x = np.delete(x,index_list,axis=0)
new_y = np.delete(y,index_list,axis=0)
    
report_path = 'D:\Study\_Report\\'#'D\\Study\_Report\\
file_name = os.path.abspath( __file__ ).split('\\')[3].split('.')[0]
file = open(report_path + file_name +".txt", "w")

model = XGBClassifier(n_jobs = -1)
x_train, x_test, y_train, y_test = train_test_split(new_x, new_y, 
        train_size = 0.8, shuffle = True, random_state = 66
        , stratify= new_y)
msg = '========== SMOTE 적용 전 ==============\n'
print(msg)  
file.write(msg)
msg = pd.Series(y).value_counts()
print(msg)  
file.write(f'{msg}\n')   

model.fit(x_train,y_train)
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

msg ='========== SMOTE 적용 ==============\n'   
print(msg)  
file.write(msg)

smote = SMOTE(random_state=66)
x_train, y_train = smote.fit_resample(x_train, y_train)

msg = pd.Series(y_train).value_counts()
print(msg)  
file.write(f'{msg}\n')   

model.fit(x_train,y_train)
msg = f'SMOTE:Model Score : {score} \n'
print(msg)  
file.write(msg)
    

y_pred = model.predict(x_test)
msg = f'SMOTE:Acc_score  {round(accuracy_score(y_test,y_pred),4)} \n'
print(msg)  
file.write(msg)

msg = f'SMOTE:f1_score : {round(f1_score(y_test,y_pred,average="macro"),4)} \n'
print(msg)  
file.write(msg)
    
file.close()
'''
========== SMOTE 적용 전 ==============
1    357
0    212
:Model Score : 0.9681 
:Acc_score  0.9681
:f1_score : 0.9532
========== SMOTE 적용 ==============
0    285
1    285
SMOTE:Model Score : 0.9681 
SMOTE:Acc_score  0.9574
SMOTE:f1_score : 0.9407
'''