import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.8125, shuffle = True, random_state = 66) 

# x_train_, x_val, y_train_, y_val = train_test_split(x_train, y_train, 
#          train_size = 0.769231, shuffle = True, random_state = 66)

#2. 모델구성
model =Sequential()
model.add(Dense(10, input_dim = 1))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs= 100, batch_size=1, validation_split=0.23)

#4 평가 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict([17])
print("17의 예측값 : ", y_predict)

'''
10/10 [==============================] - 0s 2ms/step - loss: 8.7375e-07 - val_loss: 4.2511e-07
1/1 [==============================] - 0s 93ms/step - loss: 8.9252e-07
loss :  8.925166525841632e-07
17의 예측값 :  [[16.999117]]
'''



