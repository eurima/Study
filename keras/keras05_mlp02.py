import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10], 
              [1,1.1,1.2,1.3,1.4,1.5,1.6,1.5,1.4,1.3],
              [10,9,8,7,6,5,4,3,2,1]])

y = np.array([11,12,13,14,15,16,17,18,19,20])

x = np.transpose(x) 
# x=x.T
print(x)
print(x.shape) #(10,3)
print(y.shape) #(10,)

no = [300,200,250,300,250,200,150,100,2]
model = Sequential() #Sequential 클래스의 인스턴스
model.add(Dense(no[0], input_dim = 3)) 
model.add(Dense(no[1])) 
model.add(Dense(no[2]))
model.add(Dense(no[3])) 
model.add(Dense(no[4])) 
model.add(Dense(no[5])) 
model.add(Dense(no[6])) 
model.add(Dense(no[7])) 
model.add(Dense(no[8])) 
model.add(Dense(1)) 


#3 컴파일
model.compile(loss='mse', optimizer='adam') 
epoch = 500
model.fit(x, y, epochs = epoch, batch_size=1) 
#4 평가, 예측
loss =model.evaluate(x,y)
print('loss : ', loss)
y_predict  = model.predict([[10,1.3,1]]) #x 인풋디멘션과 맞춰야! (열맞춰~!!!) 열우선 행무시
print('[10,1.3,1] 의 예측값 : ',y_predict[0][0])
print('히든 레이어 구조 :',no)

'''
loss :  0.00515368627384305
[10,1.3,1] 의 예측값 :  19.914051
히든 레이어 구조 : [300, 200, 250, 300, 250, 200, 100, 50, 2]

loss :  3.481545607364467e-10
[10,1.3,1] 의 예측값 :  19.999989
히든 레이어 구조 : [300, 200, 250, 300, 250, 200, 150, 100, 2]
'''