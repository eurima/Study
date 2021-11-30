import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time

path = "./_data/titanic/"
train = pd.read_csv(path +"train.csv", index_col = 0, header = 0)


test = pd.read_csv(path+"test.csv", index_col = 0, header = 0) #### 제출용 test는 시험용!
gender_submission = pd.read_csv(path+"gender_submission.csv", index_col = 0, header = 0) #제출할 값
# test 돌려서 나온값을 gender_submission 에 넣어서 제출한다
# print(train.shape) #(891,11)
# print(test.shape) #(418,10)
# print(gender_submission.shape) #(418,1)
# print(gender_submission).
'''
     PassengerId  Survived
0            892         0
417         1309         0
'''
'''
print(train.info())
Data columns (total 11 columns):
 #   Column    Non-Null Count  Dtype
---  ------    --------------  -----
 0   Survived  891 non-null    int64
 1   Pclass    891 non-null    int64
 2   Name      891 non-null    object
 3   Sex       891 non-null    object
 4   Age       714 non-null    float64
 5   SibSp     891 non-null    int64
 6   Parch     891 non-null    int64
 7   Ticket    891 non-null    object
 8   Fare      891 non-null    float64
 9   Cabin     204 non-null    object
 10  Embarked  889 non-null    object
dtypes: float64(2), int64(4), object(5)

print(train.describe())  ====> object 는 안나온다.

         Survived      Pclass         Age       SibSp       Parch        Fare
count  891.000000  891.000000  714.000000  891.000000  891.000000  891.000000
mean     0.383838    2.308642   29.699118    0.523008    0.381594   32.204208
std      0.486592    0.836071   14.526497    1.102743    0.806057   49.693429
min      0.000000    1.000000    0.420000    0.000000    0.000000    0.000000
25%      0.000000    2.000000   20.125000    0.000000    0.000000    7.910400
50%      0.000000    3.000000   28.000000    0.000000    0.000000   14.454200
75%      1.000000    3.000000   38.000000    1.000000    0.000000   31.000000
max      1.000000    3.000000   80.000000    8.000000    6.000000  512.329200
'''