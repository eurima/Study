import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.8125, shuffle = True, random_state = 66) #랜덤난수 고정


x_val = x_train[10:]
x_train_ = x_train[:10]

y_val = y_train[10:]
y_train_ = y_train[:10]


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
val_loss: 0.0033
loss :  0.006892136763781309
17의 예측값 :  [[16.925087]]
x_val : [ 8 13  5]
x_train_: [16  9  1 10  3 15  6 12 11  4]
x_test : [ 7  2 14]
y_val: [ 8 13  5]
y_train_: [16  9  1 10  3 15  6 12 11  4]
y_test : [ 7  2 14]

 val_loss: 0.0010
loss :  0.0020970820914953947
17의 예측값 :  [[17.04015]]    <--------------------------
x_val : [ 8 13  5]
x_train_: [16  9  1 10  3 15  6 12 11  4]
x_test : [ 7  2 14]
y_val: [ 8 13  5]
y_train_: [16  9  1 10  3 15  6 12 11  4]
y_test : [ 7  2 14]

10,8,4,2,1
'''



