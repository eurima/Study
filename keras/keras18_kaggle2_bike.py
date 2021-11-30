'''
과제 : 중위값과 평균값의 비교분석

중앙값(中央-, 영어: median) 또는 중위수(中位數)는 어떤 주어진 값들을 크기의 순서대로 정렬했을 때 가장 중앙에 위치하는 값을 의미한다. 예를 들어 1, 2, 100의 세 값이 있을 때, 2가 가장 중앙에 있기 때문에 2가 중앙값이다.

값이 짝수개일 때에는 중앙값이 유일하지 않고 두 개가 될 수도 있다. 이 경우 그 두 값의 평균을 취한다. 예를 들어 1, 10, 90, 200 네 수의 중앙값은 10과 90의 평균인 50이 된다.[1]

중앙값(median)은 중심경향치(center tendency)의 하나로 전체 데이터 중 가운데에 있는 수치 값이다. 직원이 100명인 회사에서 직원들 연봉 평균은 5천만원인데 사장의 연봉이 100억인 경우, 회사 전체의 연봉 평균은 1억 4851만 원이다. 이처럼 극단적인 값이 있다면 중앙값이 평균값보다 유용하다.


자료 집합에 대한 평균은 단순히 모든 관측값을 더해서 관측값 개수로 나눈 것이다. 일단 자료 집합의 공통성을 이렇게 설명하기로 하면, 관측값이 어떻게 다른지 설명하는 데는 보통 표준편차를 쓴다. 표준편차는 편차들(deviations)의 제곱합(SS)을 평균한 값의 제곱근이다.

'''
import time
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

import matplotlib.pyplot as plt

def RMSE(y_test, y_predict):
      return np.sqrt(mean_squared_error(y_test, y_predict))      
      

path = "./_data/bike/"
train = pd.read_csv(path +"train.csv")
# print(train) 10886,12

test_flie = pd.read_csv(path + "test.csv") #### 제출용 test는 시험용!
# print(test) 6493,9
submission = pd.read_csv(path+"sampleSubmission.csv") #제출할 값
# print(submission) 6493,2





'''
print(train.info()) ★

RangeIndex: 10886 entries, 0 to 10885
Data columns (total 12 columns):
 #   Column      Non-Null Count  Dtype
---  ------      --------------  -----
 0   datetime    10886 non-null  object =====> 지금은 문자열 취급을 해라(원래는 datetime형으로 만들어야 한다)
 1   season      10886 non-null  int64
 2   holiday     10886 non-null  int64
 3   workingday  10886 non-null  int64
 4   weather     10886 non-null  int64
 5   temp        10886 non-null  float64
 6   atemp       10886 non-null  float64
 7   humidity    10886 non-null  int64
 8   windspeed   10886 non-null  float64
 9   casual      10886 non-null  int64
 10  registered  10886 non-null  int64
 11  count       10886 non-null  int64
dtypes: float64(3), int64(8), object(1)

print(train.describe()) ★

             season       holiday    workingday       weather  ...     windspeed        casual    registered         count
count  10886.000000  10886.000000  10886.000000  10886.000000  ...  10886.000000  10886.000000  10886.000000  10886.000000
mean       2.506614      0.028569      0.680875      1.418427  ...     12.799395     36.021955    155.552177    191.574132
std        1.116174      0.166599      0.466159      0.633839  ...      8.164537     49.960477    151.039033    181.144454
min        1.000000      0.000000      0.000000      1.000000  ...      0.000000      0.000000      0.000000      1.000000
25%        2.000000      0.000000      0.000000      1.000000  ...      7.001500      4.000000     36.000000     42.000000
50%        3.000000      0.000000      1.000000      1.000000  ...     12.998000     17.000000    118.000000    145.000000
75%        4.000000      0.000000      1.000000      2.000000  ...     16.997900     49.000000    222.000000    284.000000
max        4.000000      1.000000      1.000000      4.000000  ...     56.996900    367.000000    886.000000    977.000000


print(train.columns)
Index(['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp',
       'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count'],
      dtype='object')

'''
y = train['count']
x = train.drop(['datetime','casual','registered','count'], axis =1) #
test_flie = test_flie.drop(['datetime'], axis =1) #
# print(x.columns) 
# print(y_train.shape) 10886,
# print(x.shape) #10886, 8.

# plt.plot(y)
# plt.show()  ========> 데이터 분포가 넓다면 로그를 씌우자 ex) 0 ~ 1000 : -> 1 ~ 3 ※log0은 에러! 무조건 로그변환 전 1을 더해준다

# 로그변환
y = np.log1p(y) #----------> 로그변환 하면서 +1 해줌.

x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.8, shuffle = True, random_state = 66) #

#2 모델구성#        
deep_len = [100, 80, 60, 50, 40, 50, 60, 70, 80, 100, 50, 40, 30, 20, 10, 2]
model = Sequential()
model.add(Dense(deep_len[0], input_dim =x.shape[1]))
model.add(Dense(deep_len[1], activation = 'linear' )) 
model.add(Dense(deep_len[2]))
model.add(Dense(deep_len[3])) 
model.add(Dense(deep_len[4])) 
model.add(Dense(deep_len[5])) 
model.add(Dense(deep_len[6])) 
model.add(Dense(deep_len[7])) 
model.add(Dense(deep_len[8])) 
model.add(Dense(deep_len[9])) 
model.add(Dense(deep_len[10])) 
model.add(Dense(deep_len[11])) 
model.add(Dense(deep_len[12])) 
model.add(Dense(deep_len[13])) 
model.add(Dense(deep_len[14])) 
model.add(Dense(deep_len[15])) 
model.add(Dense(1))

#3. 컴파일, 훈련
epoch = 10000
model.compile(loss = "mse", optimizer = 'adam') 

from tensorflow.keras.callbacks import EarlyStopping
patience_num = 50
es = EarlyStopping(monitor='val_loss', patience=patience_num, mode = 'auto', verbose=1, restore_best_weights=True)
start = time.time()

model.fit(x_train, y_train, epochs = epoch, validation_split=0.2,callbacks=[es])#batch_size =32 가 default
end = time.time() - start
print('시간 : ', round(end,2) ,'초')

#4 평가예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss) 

y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)

rmse = RMSE(y_test, y_predict)
print("R2 : ", r2)
print("RMSE : ",rmse)
print("epochs :",epoch)

'''
log1p 사용시

Epoch 00076: early stopping
시간 :  21.0 초
69/69 [==============================] - 0s 507us/step - loss: 1.4550
loss :  1.4550199508666992
R2 :  0.2574743161192756
RMSE :  1.2062418943647923 -----> 로그화 되어있으므로 RMSLE와 유사하다

'''

############### 제출용
result = model.predict(test_flie)
submission['count'] = result

# print(submission[:10])
submission.to_csv(path+"sampleSubmission_.csv", index = False)

