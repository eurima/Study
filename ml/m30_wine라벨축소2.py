import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score

path = "D:\\_data\\"
dataset = pd.read_csv(path + "winequality-white.csv",index_col=None, header=0, sep=';')

dataset = dataset.values #Numpy 자료형으로 변환
x = dataset[:,:11] # : 모든행 , 0 ~ 11번째행까지
y = dataset[:,11] # : 모든행 , 11번째행만
for index, value in enumerate(y):
    if value == 9:
        y[index] = 7
    elif value == 8:
        y[index] = 7
    else:
        y[index] = value

# print(np.unique(y,return_counts=True))        
print(pd.Series(y).value_counts())

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.8, shuffle = True, random_state = 66
         , stratify= y)

print('========== SMOTE 적용 ==============')

smote = SMOTE(random_state=66, k_neighbors=3)
x_train, y_train = smote.fit_resample(x_train, y_train)
print(pd.Series(y_train).value_counts())

model = XGBClassifier(n_jobs = -1)
model.fit(x_train,y_train)

score = model.score(x_test,y_test)
print(':Model Score : ', round(score,4))

y_pred = model.predict(x_test)
print(':Acc_score : ',round(accuracy_score(y_test,y_pred),4))

print(':f1_score : ',round(f1_score(y_test,y_pred,average='macro'),4))        