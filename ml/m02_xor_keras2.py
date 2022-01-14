import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#1. 데이터
x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [0,1,1,0]

#2. 모델
model = Sequential()
model.add(Dense(100, input_dim =2, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, batch_size=1, epochs=10)

#4. 평가 예측
y_predict = model.predict(x_data)
num = 0
result = model.evaluate(x_data,y_data)
print("model_acc : ",result[1])
print(x_data,"의 예측 결과 : ", int(y_predict[0]),int(y_predict[1]),int(y_predict[2]),int(y_predict[3]))

'''
model_acc :  1.0
[[0, 0], [0, 1], [1, 0], [1, 1]] 의 예측 결과 :  [[1.3682370e-03]
 [9.9807346e-01]
 [9.9741805e-01]
 [8.7732030e-04]]
'''