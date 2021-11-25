import numpy as np
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x_train = np.array([1,2,3,4,5,6,7])
x_test = np.array([8,9,10])
y_train = np.array([1,2,3,4,5,6,7])
y_test = np.array([8,9,10])

#2. 모델
no = [200,150,100,50,100,150,100,50,10]
model = Sequential() 
model.add(Dense(no[0], input_dim = 1)) 
model.add(Dense(no[1])) 
model.add(Dense(no[2]))
model.add(Dense(no[3])) 
model.add(Dense(no[4])) 
model.add(Dense(no[5])) 
model.add(Dense(no[6])) 
model.add(Dense(no[7])) 
model.add(Dense(no[8])) 
model.add(Dense(1)) 

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('Loss = ',loss)
res =[11]
result = model.predict(res)
print(f'{res}의 예측값 :',result)

