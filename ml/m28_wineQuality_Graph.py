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

# dataset = dataset.values #Numpy 자료형으로 변환
# x = dataset[:,:11] # : 모든행 , 0 ~ 11번째행까지
# y = dataset[:,11] # : 모든행 , 11번째행만
x = dataset.drop('quality', axis=1)
y = dataset.quality

##데이터의 분포
# plt.bar 
# groupby 쓰고, count() 쓰자
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')


count_data = dataset.groupby("quality")[ "quality"].count()
print(count_data)
count_data.plot(kind='bar', rot=0)
# dataset.groupby( [ "quality"] ).count().plot(kind='bar', rot=0)
p = pd.DataFrame({'count' : dataset.groupby( [ "quality"] ).size()})
# print(p)
# p.plot(kind='bar', rot=0)
# p['count'].plot(kind='bar', rot=0)


plt.show()
print(p)
# print(g1)
print(dataset.index)
print(dataset.groupby(['quality']).count())#['pH'].plot(kind='bar', rot=0)
# plt.show()
