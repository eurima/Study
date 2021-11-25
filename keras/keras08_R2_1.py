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
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,9,8,12,13,17,12,14,21,14,11,19,23,25])

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
deep_len = [100,40,30,20,30,20,10,5,2]
model = Sequential() #Sequential 클래스의 인스턴스
model.add(Dense(deep_len[0], input_dim = 1)) 
model.add(Dense(deep_len[1])) 
model.add(Dense(deep_len[2]))
model.add(Dense(deep_len[3])) 
model.add(Dense(deep_len[4])) 
model.add(Dense(deep_len[5])) 
model.add(Dense(deep_len[6])) 
model.add(Dense(deep_len[7])) 
model.add(Dense(deep_len[8])) 
model.add(Dense(1)) 


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer ='adam')
model.fit(x_train, y_train, epochs = 500, batch_size =1)

#4 평가예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss)

y_predict = model.predict(x_test)
# plt.scatter(x, y)
# plt.plot(x, y_predict, color = 'red')
# plt.show()
r2 = r2_score(y_test,y_predict)
print("r2 : ",r2)
print(deep_len)

'''
loss :  9.014548301696777
r2 :  0.366164551466989
60,40,20,15,25,15,10,5,2

r2 :  0.3803399620916895
[5, 50, 20, 10, 25, 20, 10, 5, 2]

loss :  8.276556968688965
r2 :  0.4180545397914853
[100, 40, 30, 20, 30, 20, 10, 5, 2]

'''

