import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10], 
              [1,2,3,4,5,6,7,8,9,10]])

y = np.array([11,12,13,14,15,16,17,18,19,20])

#print(x.shape) #(2,10)
#print(y.shape) #(10,)

# #2 모델구성
# model = Sequential() #Sequential 클래스의 인스턴스
# model.add(Dense(5, input_dim = 2)) #출력 5, 인풋1
# model.add(Dense(3)) #출력 3 (인풋5) Sequential구조이기 때문에 인풋 별도 명시 X
# model.add(Dense(4)) #출력 4 (인풋3)
# model.add(Dense(2)) #출력 2 (인풋4)
# model.add(Dense(1)) #출력 1 (인풋2)
# #3 컴파일
# model.compile(loss='mse', optimizer='adam') 
# epoch = 100
# model.fit(x, y, epochs = epoch, batch_size=2) 
# #4 평가, 예측
# loss =model.evaluate(x,y)
# print('loss : ', loss)
# y_predict  = model.predict([[10,1.3]])
# print('[10,1.3] 의 예측값 : ',y_predict)

#ValueError: Failed to convert a NumPy array to a Tensor 
# (Unsupported object type list).