import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings(action='ignore')

path = "D:\\_data\\"
dataset = pd.read_csv(path + "winequality-white.csv",index_col=None, header=0, sep=';')

x = dataset.drop('quality', axis=1)
y = dataset.quality

file = open("T와인 라벨축소.txt", "w")

y_list=[]
y_list.append(y.apply(lambda x: 0 if x <= 4 else 1 if x<= 7 else 2))
y_list.append(y.apply(lambda x: 0 if x <= 5 else 1 if x == 6 else 2))

model = XGBClassifier(n_jobs = -1)

for y in y_list:
    x_train, x_test, y_train, y_test = train_test_split(x, y, 
            train_size = 0.8, shuffle = True, random_state = 66
            , stratify= y)
    print('========== SMOTE 적용 전 ==============')  
    file.write('========== SMOTE 적용 전 ==============\n')
    print(pd.Series(y).value_counts())  
    
    model.fit(x_train,y_train)
    score = model.score(x_test,y_test)
    print(':Model Score : ', round(score,4))
    file.write(f':Model Score : {round(score,4)} \n')    

    y_pred = model.predict(x_test)
    print(':Acc_score : ',round(accuracy_score(y_test,y_pred),4))
    print(':f1_score : ',round(f1_score(y_test,y_pred,average='macro'),4)) 
    
    print('========== SMOTE 적용 ==============')    
    smote = SMOTE(random_state=66, k_neighbors=3)
    x_train, y_train = smote.fit_resample(x_train, y_train)
    
    model.fit(x_train,y_train)
    score = model.score(x_test,y_test)
    print(':SMOTE Model Score : ', round(score,4))

    y_pred = model.predict(x_test)
    print(':SMOTE Acc_score : ',round(accuracy_score(y_test,y_pred),4))
    print(':SMOTE f1_score : ',round(f1_score(y_test,y_pred,average='macro'),4))
    
'''
:Model Score :  0.7051
:Acc_score :  0.7051
:f1_score :  0.7039
1    2198
0    1640
2    1060

:Model Score :  0.8306
:Acc_score :  0.8306
:f1_score :  0.6463
1    3655
2    1060
8     183
'''
'''
========== SMOTE 적용 전 ==============
1    4535
0     183
2     180
:Model Score :  0.9367
:Acc_score :  0.9367
:f1_score :  0.5923

========== SMOTE 적용 ==============
:SMOTE Model Score :  0.9245
:SMOTE Acc_score :  0.9245  
:SMOTE f1_score :  0.6252

========== SMOTE 적용 전 ==============
1    2198
0    1640
2    1060
========== SMOTE 적용 ==============
:SMOTE Model Score :  0.7051
:SMOTE Acc_score :  0.7051
:SMOTE f1_score :  0.7039
'''