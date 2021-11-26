import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.8125, shuffle = True, random_state = 66) #랜덤난수 고정

x_train_, x_val, y_train_, y_val = train_test_split(x_train, y_train, 
         train_size = 0.769231, shuffle = True, random_state = 66)


#2. 모델구성
model =Sequential()
model.add(Dense(10, input_dim = 1))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train_, y_train_, epochs= 100, batch_size=1, 
          validation_data= (x_val, y_val))

#4 평가 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict([17])
print("17의 예측값 : ", y_predict)

print("x_val :",x_val)
print("x_train_:",x_train_)
print("x_test :",x_test)

print("y_val:",y_val)
print("y_train_:",y_train_)
print("y_test :",y_test)

'''
loss :  2.7145782510729077e-08
17의 예측값 :  [[16.999865]]
x_val : [ 6  9 13]
x_train_: [ 5 11 16  4  1  8 15 10 12  3]
x_test : [ 7  2 14]
y_val: [ 6  9 13]
y_train_: [ 5 11 16  4  1  8 15 10 12  3]
y_test : [ 7  2 14]
'''



