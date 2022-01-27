from sklearn.metrics import accuracy_score,f1_score

from sklearn.preprocessing import PowerTransformer
from sklearn.compose import make_column_transformer

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from xgboost.plotting import plot_importance
import matplotlib.pyplot as plt
import time
import sys

path = "D:\\_data\\"
dataset = pd.read_csv(path + "winequality-white.csv",index_col=None, header=0, sep=';')
# print(dataset.shape) #(4898, 12)
# print(dataset.describe()) #수치데이터만

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
# x = dataset.drop('quality', axis=1)
# y = dataset.quality
#라벨 : (array([   3.,   4.,    5.,   6.,   7.,   8.,   9.]), 
#        array([  20,  163,  1457,  2198,   880,   175,    5], dtype=int64))

new_y = []
for i in y:
    if int(i) == 3 or int(i) == 4:
        new_y.append(4)
    elif int(i) == 8 or int(i) == 9:
        new_y.append(9)
    else:
        new_y.append(i)
# print(new_y)
new_y = np.where(y<=4,4,np.where(y>=8,9,y))
print(np.unique(new_y))


from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import FeatureUnion

union_transformer = FeatureUnion([("pca", PCA(n_components=1)), ("svd", TruncatedSVD(n_components=1))])
union_transformer.fit_transform(dataset)  
# print(union_transformer.transform(dataset))

# x = dataset.drop('quality', axis=1)
# y = dataset.quality
# print(dataset.head())

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, new_y, 
         train_size = 0.8, shuffle = True, random_state = 66, stratify= y)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# transformer = make_column_transformer(
#     (PowerTransformer(), ['library']),
#     remainder='passthrough')

# transformer = PowerTransformer()

# x_train = transformer.fit_transform(x_train)
# x_test = transformer.transform(x_test)

model = XGBClassifier()

model.fit(x_train, y_train)
score = model.score(x_test, y_test)
y_perd = model.predict(x_test)
print("Score :",score)
print("Acc :", accuracy_score(y_test,y_perd))

f1 = f1_score(y_test,y_perd, average="macro")
print("F1 : ", f1)

'''
Score : 0.6591836734693878
Acc : 0.6591836734693878
F1 :  0.41005452777318885

Score : 0.6642857142857143
Acc : 0.6642857142857143
F1 :  0.5727101289444397
'''