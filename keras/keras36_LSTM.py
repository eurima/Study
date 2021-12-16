import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.callbacks import EarlyStopping
#1 데이터
x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6]])
y = np.array([4,5,6,7])

print(x.shape, y.shape) #(4,3), (4,)

#input_shape = (batch_size, timestep, feature) (행, 열, 몇개씩 자르는지)
x = x.reshape(4,3,1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.9, shuffle = True, random_state = 66) 
scaler = StandardScaler()
# print(x)
#2 모델구성
model = Sequential()
model.add(LSTM(100,activation = 'linear', input_shape = (3,1)))
# model.add(SimpleRNN(100,activation = 'linear', input_shape = (3,1)))
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
model.compile(loss='mse', optimizer = 'adam')
es = EarlyStopping(monitor='val_loss', patience=500, mode = 'auto', restore_best_weights=True)
model.fit(x_train, y_train, epochs = 10000, validation_split=0.2, callbacks=[es],batch_size = 1)
model.evaluate(x_test,y_test)
y_ = np.array([[5,6,7],
              [6,7,8],
              [9,10,11]]).reshape(3,3,1) #------> input과 열맞춰라
result = model.predict(y_)
print(result) #8.008091  
'''
SImpleRNN
[[ 8.009016]
 [ 9.021783]
 [12.064664]] 
 [[ 7.9937906]
 [ 8.996402 ]
 [12.00195  ]]
 ----------------------
 LSTM
 [[ 7.9695635]
 [ 8.920452 ]
 [11.635194 ]]
 
 [[ 8.038758]
 [ 9.086261]
 [12.313028]]
'''

 