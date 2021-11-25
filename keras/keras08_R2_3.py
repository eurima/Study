'''
R2값을 찾아라
결정계수는 회귀모델에서 독립변수가 종속변수를 얼마나 설명해주는지를 가리키는 지표 (설명력)
'''
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

#데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.7, shuffle = True, random_state = 66) #랜덤난수 고정

#2 모델
# model = Sequential()
# model.add(Dense(20, input_dim = 1))
# model.add(Dense(10))
# model.add(Dense(5))
# model.add(Dense(10))
# model.add(Dense(8))
# model.add(Dense(4))
# model.add(Dense(1))
deep_len = [5,5,5,5,5]
model = Sequential() #Sequential 클래스의 인스턴스
model.add(Dense(deep_len[0], input_dim = 1)) 
model.add(Dense(deep_len[1])) 
model.add(Dense(deep_len[2]))
model.add(Dense(deep_len[3])) 
model.add(Dense(deep_len[4])) 
# model.add(Dense(deep_len[5])) 
# model.add(Dense(deep_len[6])) 
# model.add(Dense(deep_len[7])) 
# model.add(Dense(deep_len[8])) 
model.add(Dense(1)) 

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y, epochs = 6000, batch_size =1)

#4 평가예측
loss = model.evaluate(x,y)
print("loss : ",loss)

y_predict = model.predict(x)
r2 = r2_score(y,y_predict)
print("r2 : ",r2)
print(deep_len)

'''
r2 :  0.8099606270892693
[10, 10, 10, 10, 10]

loss :  0.3800448775291443
r2 :  0.8099774753987774
[50, 40, 30, 40, 50]

'''

