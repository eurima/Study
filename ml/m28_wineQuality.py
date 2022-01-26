from sklearn.metrics import accuracy_score,f1_score
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from xgboost.plotting import plot_importance
import matplotlib.pyplot as plt
import time

path = "D:\\_data\\"
dataset = pd.read_csv(path + "winequality-white.csv",index_col=None, header=0, sep=';')
# print(dataset.shape) #(4898, 12)
# print(dataset.describe()) #수치데이터만
'''
       fixed acidity  volatile acidity  citric acid  ...    sulphates      alcohol      quality
count    4898.000000       4898.000000  4898.000000  ...  4898.000000  4898.000000  4898.000000
mean        6.854788          0.278241     0.334192  ...     0.489847    10.514267     5.877909
std         0.843868          0.100795     0.121020  ...     0.114126     1.230621     0.885639
min         3.800000          0.080000     0.000000  ...     0.220000     8.000000     3.000000
25%         6.300000          0.210000     0.270000  ...     0.410000     9.500000     5.000000
50%         6.800000          0.260000     0.320000  ...     0.470000    10.400000     6.000000
75%         7.300000          0.320000     0.390000  ...     0.550000    11.400000     6.000000
max        14.200000          1.100000     1.660000  ...     1.080000    14.200000     9.000000
'''
# print(dataset.info()) 
'''
 #   Column                Non-Null Count  Dtype
---  ------                --------------  -----
 0   fixed acidity         4898 non-null   float64
 1   volatile acidity      4898 non-null   float64
 2   citric acid           4898 non-null   float64
 3   residual sugar        4898 non-null   float64
 4   chlorides             4898 non-null   float64
 5   free sulfur dioxide   4898 non-null   float64
 6   total sulfur dioxide  4898 non-null   float64
 7   density               4898 non-null   float64
 8   pH                    4898 non-null   float64
 9   sulphates             4898 non-null   float64
 10  alcohol               4898 non-null   float64
 11  quality               4898 non-null   int64
'''
dataset = dataset.values #Numpy 자료형으로 변환
x = dataset[:,:11] # : 모든행 , 0 ~ 11번째행까지
y = dataset[:,11] # : 모든행 , 11번째행만

print("라벨 :",np.unique(y, return_counts = True))
#라벨 : (array([3., 4., 5., 6., 7., 8., 9.]), 
#        array([  20,  163, 1457, 2198,  880,  175,    5], dtype=int64))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.8, shuffle = True, random_state = 66, stratify= y)

# 주의~!!!! stratify의 y는 yes가 아니라 y = dataset[:,11]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = XGBClassifier()

model.fit(x_train, y_train)
score = model.score(x_test, y_test)
y_perd = model.predict(x_test)
print("Score :",score)
print("Acc :", accuracy_score(y_test,y_perd))

f1 = f1_score(y_test,y_perd, average="macro")
print("F1 : ", f1)


