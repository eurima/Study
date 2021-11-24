import numpy as np
from numpy.core.fromnumeric import transpose
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([range(10), range(21,31), range(201,211)])
# print(x)
x = transpose(x)
# print(x.shape) #(10,3)

y = np.array([[1,2,3,4,5,6,7,8,9,10], 
              [1,1.1,1.2,1.3,1.4,1.5,1.6,1.5,1.4,1.3],
              [10,9,8,7,6,5,4,3,2,1]])

y = transpose(y)
# print(y.shape) #(10,3)

no = [300,250,200,150,200,100,50,10,3]
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
model.add(Dense(3)) 


#3 컴파일
model.compile(loss='mse', optimizer='adam') #optimizer 최적화 함수 / loss 손실함수
epoch = 500
model.fit(x, y, epochs = epoch, batch_size=1) 
#4 평가, 예측
loss =model.evaluate(x,y)
print('loss : ', loss)
y_prophet = [[9,30,210]] #<== 예측할 값
y_predict  = model.predict(y_prophet) 
print(f'{y_prophet} 의 예측값 : ',y_predict[0])
print('히든 레이어 구조 :',no) #10,1.3,1

'''
loss :  1.8106825351715088
[9,30,210] 의 예측값 :  [[7.272273  4.452197  1.9547926]]
히든 레이어 구조 : [300, 200, 250, 300, 250, 200, 150, 100, 2]

loss :  0.01771879568696022
[9,30,210] 의 예측값 :  [[10.061534    1.546223    0.72618777]]
히든 레이어 구조 : [300, 200, 250, 300, 250, 200, 150, 100, 3]

loss :  0.02837589755654335
[9,30,210] 의 예측값 :  [10.031609   1.2052634  1.0042027]
히든 레이어 구조 : [200, 150, 100, 150, 200, 150, 100, 50, 3]

loss :  0.33651408553123474
[9,30,210] 의 예측값 :  [10.561804    1.751431    0.68401396]
히든 레이어 구조 : [150, 100, 120, 100, 80, 50, 40, 10, 3]

loss :  0.036481425166130066
[9,30,210] 의 예측값 :  [9.812468  1.4139774 1.1363528]
히든 레이어 구조 : [300, 250, 200, 150, 200, 100, 50, 25, 3]

loss :  0.009634817950427532 <-------------------------
[9,30,210] 의 예측값 :  [10.037094   1.4233235  0.9670598]
히든 레이어 구조 : [300, 250, 200, 150, 200, 100, 50, 10, 3]

'''