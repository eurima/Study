import pandas as pd
import time
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import fetch_california_housing, load_boston, load_diabetes
from sklearn.model_selection import learning_curve, train_test_split, GridSearchCV

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, accuracy_score
import warnings
# warnings.filterwarnings('ignore')

'''
1. QuantileTransformer : 지정된 분위수에 맞게 데이터를 변환한다. 기본 분위수는 1,000개이며 n_quantiles 매개변수에서 변경할 수 있음
2. PowerTransformer : 
'''

#1. 데이터

datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=66)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#훈련
model = XGBRegressor(
    n_estimators=10000,       
    learning_rate = 0.01,    
    gamma = 1,
    # 트리에서 가지를 추가로 치기 위해 필요한 최소한의 손실 감소 기준. 기준값이 클 수록 모형이 더 단순해진다.(> 0)    
    max_depth = 20,
    #트리의 최대 깊이.(> 0)
    min_child_weight = 5,
    #트리에서 가지를 추가로 치기 위해 필요한 최소한의 사례 수.(> 0)
    
    ###############    규제
    reg_lambda = 1,
    # 기본값 : 1 / 범위 : [0,∞]
    # L2 정규화(규제) 파라미터이다.
    # 커질수록 보수적인 모델을 생성하고 오버 피팅을 방지해준다. 지나치게 클 경우 언더 피팅이 난다. 
    # 너무 큰 가중치를 그 크기에 비례하여 줄여준다.
    # noise나 outlier 같은 애들이나 너무 크게 튀는 데이터들을 어느 정도 잡아준다고 보면 된다. 
    # gamma, alpha와 함께 튜닝함.    
    reg_alpha = 0,
    # 기본값 : 0 / 범위: [0,∞]
    # L1 정규화(규제) 파라미터이다.
    # 커질수록 보수적인 모델을 생성하고 오버 피팅을 방지해준다.
    # 불필요한 가중치를 0으로 만들어서 무시하도록 한다. 
    # sparse feature 가 있거나 feature수가 지나치게 많을 때 효과적이다.
    # gamma, lambda와 함께 튜닝함.    
    n_jobs=-1 
)
# model = XGBRegressor()

start = time.time()
model.fit(x_train, y_train, verbose=1)
end = time.time()-start
print("걸린 시간 : ", round(end, 2))

score = round(model.score(x_test, y_test),2)
print("score : ", score)

y_pred = model.predict(x_test)
r2 = round(r2_score(y_test, y_pred),2)
print("r2 : ", r2)
'''
    n_estimators=5000,       # 0.8441118702517121
    learning_rate = learningRate,    # 0.8531073618718206
    gamma = 1,
    # 트리에서 가지를 추가로 치기 위해 필요한 최소한의 손실 감소 기준. 기준값이 클 수록 모형이 더 단순해진다.(> 0)
    
    max_depth = 3,
    #트리의 최대 깊이.(> 0)
    min_child_weight = 1,
    #트리에서 가지를 추가로 치기 위해 필요한 최소한의 사례 수.(> 0)
    
    #규제
    reg_lambda = 1,
    # 기본값 : 1 / 범위 : [0,∞]
    # L2 정규화(규제) 파라미터이다.
    # 커질수록 보수적인 모델을 생성하고 오버 피팅을 방지해준다. 지나치게 클 경우 언더 피팅이 난다. 
    # 너무 큰 가중치를 그 크기에 비례하여 줄여준다.
    # noise나 outlier 같은 애들이나 너무 크게 튀는 데이터들을 어느 정도 잡아준다고 보면 된다. 
    # gamma, alpha와 함께 튜닝함.    
    reg_alpha = 0,
    # 기본값 : 0 / 범위: [0,∞]
    # L1 정규화(규제) 파라미터이다.
    # 커질수록 보수적인 모델을 생성하고 오버 피팅을 방지해준다.
    # 불필요한 가중치를 0으로 만들어서 무시하도록 한다. 
    # sparse feature 가 있거나 feature수가 지나치게 많을 때 효과적이다.
    # gamma, lambda와 함께 튜닝함.    
    n_jobs=-1 

걸린 시간 :  2.4
score :  0.9424608425289078
r2 :  0.9424608425289078
learningRate : 0.0545
'''