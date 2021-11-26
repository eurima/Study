'''
val_loss 의 값을 더 신뢰를 해야 한다

'''
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))

x_train = x[:11]
y_train = y[:11]
x_test = x[11:14]
y_test = y[11:14]
x_val = x[14:]
y_val = y[14:]

#2. 모델구성
model =Sequential()
model.add(Dense(10, input_dim = 1))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs= 100, batch_size=1, 
          validation_data= (x_val, y_val))

#4 평가 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict([17])
print("17의 예측값 : ", y_predict)


'''
val_loss: 0.0120
loss :  0.004829872399568558
17의 예측값 :  [[16.86399]]

- val_loss: 7.3615e-04
loss :  0.0003065388882532716
17의 예측값 :  [[16.966534]]
10,8,4,2,1

loss :  9.38898665481247e-08
17의 예측값 :  [[16.999418]]
x_test [11 12 13]
x_train [ 0  1  2  3  4  5  6  7  8  9 10]
x_val [14 15 16]
'''
print("x_test",x_test)
print("x_train",x_train)
print("x_val",x_val)
