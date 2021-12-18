import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping
import time
import datetime
date = datetime.datetime.now()
#1 데이터
x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9],
              [8,9,10],
              [9,10,11],
              [10,11,12],
              [20,30,40],
              [30,40,50],
              [40,50,60]              
              ])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
y_predict = np.array([50,60,70]) 
y_predict= y_predict.reshape(1, 3)

# print(x.shape, y.shape) #(13,3) (13,)
#input_shape = (batch_size, timestep, feature) (행, 열, 몇개씩 자르는지)
x = x.reshape(13,3,1)
y_predict = y_predict.reshape(1,3,1)
#2 모델구성
model = Sequential()
model.add(LSTM(150,return_sequences= True,activation = 'relu', input_shape = (3,1)))
model.add(LSTM(100,return_sequences= True, activation = 'relu'))
model.add(LSTM(80,return_sequences= True, activation = 'relu'))
model.add(LSTM(60,return_sequences= True, activation = 'relu'))
model.add(LSTM(40,return_sequences= True, activation = 'relu'))
model.add(LSTM(60,return_sequences= True, activation = 'relu'))
model.add(LSTM(80, activation = 'relu'))
model.add(Dense(80, activation = 'relu'))
model.add(Dense(70, activation = 'relu'))
model.add(Dense(60, activation = 'relu'))
model.add(Dense(80, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(80, activation = 'relu'))
model.add(Dense(60, activation = 'relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(20, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1))
#3 컴파일
start = time.time()
model.compile(loss='mse', optimizer = 'adam',metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=50, mode = 'auto', restore_best_weights=True)
model.fit(x, y, epochs = 1000, validation_split=0.2,callbacks=[es], batch_size = 1)

result = model.predict(y_predict)
print(result) #[[81.6288]]
end = time.time() - start
print('시간 : ', round(end,2) ,'초')
'''
LSTM
[[89.8557]]
시간 :  2.78 초
GRU
[[69.0053]]
시간 :  2.64 초
[[102.231384]]
시간 :  9.44 초
'''




 