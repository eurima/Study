import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from imblearn.over_sampling import SMOTE

datasets = load_wine()
x = datasets.data
y = datasets.target
# print(x.shape, y.shape) #(178, 13) (178,)
# print(pd.Series(y).value_counts())
# 1    71
# 0    59
# 2    48

x_new = x[:-30]
y_new = y[:-30]
# print(pd.Series(y_new).value_counts())

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_new, y_new, 
         train_size = 0.8, shuffle = True, random_state = 66, stratify= y_new)

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

model = XGBClassifier(n_jobs = -1)
model.fit(x_train,y_train)

score = model.score(x_test,y_test)
print(':불균형 상태 Model Score : ', round(score,4))

y_pred = model.predict(x_test)
print(':불균형 상태 Acc_score : ',round(accuracy_score(y_test,y_pred),4))

print('========== SMOTE 적용 ==============')

smote = SMOTE(random_state=66)
x_train, y_train = smote.fit_resample(x_train, y_train)

model.fit(x_train,y_train)

score = model.score(x_test,y_test)
print(':Model Score : ', round(score,4))

y_pred = model.predict(x_test)
print(':Acc_score : ',round(accuracy_score(y_test,y_pred),4))

      