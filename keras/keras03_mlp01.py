import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10], 
              [1,1.1,1.2,1.3,1.4,1.5,1.6,1.5,1.4,1.3]])

y = np.array([11,12,13,14,15,16,17,18,19,20])

#x = x.reshape(10,2) # <==== (2,10) 을 (10,2) 로 바꿔준다
# print(x)
# print(x.shape)
x = np.transpose(x) 
# x=x.T
print(x)
# print(x.shape) #(2,10)
# #print(y.shape) #(10,)


no = [300,200,250,300,250,200,100,50,2]
model = Sequential() #Sequential 클래스의 인스턴스
model.add(Dense(no[0], input_dim = 2)) 
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
y_predict  = model.predict([[10,1.3]]) #x 인풋디멘션과 맞춰야! (열맞춰~!!!) 열우선 행무시
print('[10,1.3] 의 예측값 : ',y_predict[0][0])
print('히든 레이어 구조 :',no)

'''
loss :  0.9085286855697632
[10,1.3] 의 예측값 :  9.50081
히든 레이어 구조 : [10, 10, 10, 10, 10, 10, 10, 10, 2]

loss :  0.0010585206327959895
[10,1.3] 의 예측값 :  10.94503
히든 레이어 구조 : [50, 40, 30, 20, 10, 5, 4, 3, 2]

loss :  0.003381619928404689
[10,1.3] 의 예측값 :  10.990032
히든 레이어 구조 : [50, 40, 50, 40, 20, 10, 5, 4, 2]

loss :  0.00018545889179222286
[10,1.3] 의 예측값 :  10.975661
히든 레이어 구조 : [300, 200, 100, 50, 40, 20, 10, 5, 2]

loss :  0.07099814713001251
[10,1.3] 의 예측값 :  19.595469
히든 레이어 구조 : [300, 200, 250, 300, 250, 200, 100, 50, 2]
x=x.T

x = np.transpose(x) 
epoch=500, batch_size=1
loss :  0.00011681378964567557
[10,1.3] 의 예측값 :  19.98447
히든 레이어 구조 : [300, 200, 250, 300, 250, 200, 100, 50, 2]
'''
