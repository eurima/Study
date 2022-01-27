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

#1 0dmf 112개 삭제

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

report_path = 'D\\Study\_Report\\'
file_name = os.path.abspath( __file__ ).split('\\')[3].split('.')[0]
file = open(report_path + file_name +".txt", "w")

# y_list=[]
# y_list.append(y.apply(lambda x: 0 if x <= 4 else 1 if x<= 7 else 2))
# y_list.append(y.apply(lambda x: 0 if x <= 5 else 1 if x == 6 else 2))

model = XGBClassifier(n_jobs = -1)
# for y in y_list:
x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size = 0.8, shuffle = True, random_state = 66
        , stratify= y)
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

msg = f':f1_score : ,{round(f1_score(y_test,y_pred,average="macro"),4)} \n'
print(msg)  
file.write(msg)

msg ='========== SMOTE 적용 ==============\n'   
print(msg)  
file.write(msg)


smote = SMOTE(random_state=66, k_neighbors=3)
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

msg = f'SMOTE:f1_score : ,{round(f1_score(y_test,y_pred,average="macro"),4)} \n'
print(msg)  
file.write(msg)
    
file.close()