from re import X
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

# 출력 관련 옵션
# np.set_printoptions(threshold=sys.maxsize)
# pd.set_option('display.max_row',100)
# pd.set_option('display.max_columns',50)
# pd.set_option('display.width', 190)

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
# dataset = dataset.values #Numpy 자료형으로 변환
# x = dataset[:,:11] # : 모든행 , 0 ~ 11번째행까지
# y = dataset[:,11] # : 모든행 , 11번째행만
x = dataset.drop('quality', axis=1)
y = dataset.quality

# print("라벨 :",np.unique(y, return_counts = True))
#라벨 : (array([3., 4., 5., 6., 7., 8., 9.]), 
#        array([  20,  163, 1457, 2198,  880,  175,    5], dtype=int64))

### 아웃라이어 확인 ####
from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=.1)
def out_mean(x):
    for i in range(11):
        col = x[:,i].reshape(-1,1)
        outliers.fit(col)
        
        for c in col:  
            print(c)  
            Ol = outliers.predict(col)    
            # Ol = outliers.predict(c)
            # if Ol == -1:        
            #     print(Ol)
            # else:
            #     print(x[i])
        # Ol 안에서 bool자든 뭘 이용해서 어떤조건이 == -1 이면 그 자리의 -1에 평균값 
        break
    
def boxplot_vis(data, target_name):
    plt.figure(figsize=(30, 30))
    for col_idx in range(len(data.columns)):
        # 6행 2열 서브플롯에 각 feature 박스플롯 시각화
        plt.subplot(6, 2, col_idx+1)
        # flierprops: 빨간색 다이아몬드 모양으로 아웃라이어 시각화
        plt.boxplot(data[data.columns[col_idx]], flierprops = dict(markerfacecolor = 'r', marker = 'D'))
        # 그래프 타이틀: feature name
        plt.title("Feature" + "(" + target_name + "):" + data.columns[col_idx], fontsize = 20)
    # plt.savefig('../figure/boxplot_' + target_name + '.png')
    plt.show()    
# boxplot_vis(x,'white_wine')

def df_remove_outlier(input_data, iqr_val):
    q1 = input_data.quantile(0.25) # 제 1사분위수
    q3 = input_data.quantile(0.75) # 제 3사분위수
    iqr = q3 - q1 # IQR(Interquartile range) 계산
    minimum = q1 - (iqr * iqr_val) # IQR 최솟값
    maximum = q3 + (iqr * iqr_val) # IQR 최댓값
    # IQR 범위 내에 있는 데이터만 산출(IQR 범위 밖의 데이터는 이상치)
    df_removed_outlier = input_data[(minimum < input_data) & (input_data < maximum)]
    df_outlier = input_data[(minimum > input_data) | (input_data > maximum)]
    return df_outlier
    # return df_removed_outlier

print(df_remove_outlier(x, iqr_val = 1.1))
### 아웃라이어 처리 ###
# out_mean(x)



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.8, shuffle = True, random_state = 66, stratify= y)

# 주의~!!!! stratify의 y는 yes가 아니라 y = dataset[:,11]

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# transformer = make_column_transformer(
#     (PowerTransformer(), ['library']),
#     remainder='passthrough')

# transformer = PowerTransformer()

# x_train = transformer.fit_transform(x_train)
# x_test = transformer.transform(x_test)

# model = XGBClassifier()

# model.fit(x_train, y_train)
# score = model.score(x_test, y_test)
# y_perd = model.predict(x_test)
# print("Score :",score)
# print("Acc :", accuracy_score(y_test,y_perd))

# f1 = f1_score(y_test,y_perd, average="macro")
# print("F1 : ", f1)
'''
제거 후
x = remove_outlier(x)
Score : 0.6602040816326531
Acc : 0.6602040816326531
F1 :  0.4022000074819726

out_mean(x)
Score : 0.6591836734693878
Acc : 0.6591836734693878
F1 :  0.41005452777318885

제거전
Score : 0.6591836734693878
Acc : 0.6591836734693878
F1 :  0.41005452777318885

PowerTransformer
Score : 0.6571428571428571
Acc : 0.6571428571428571
F1 :  0.4051231039101659
'''


