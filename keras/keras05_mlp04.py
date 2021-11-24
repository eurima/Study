import numpy as np
from numpy.core.fromnumeric import transpose
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([range(10)])
# print(x)
x = transpose(x)
print(x.shape) #(10,) <----(10,1) 과 같이 판단한다

y = np.array([[1,2,3,4,5,6,7,8,9,10], 
              [1,1.1,1.2,1.3,1.4,1.5,1.6,1.5,1.4,1.3],
              [10,9,8,7,6,5,4,3,2,1]])

y = transpose(y)
print(y.shape) #(10,3)

no = [200,150,100,50,100,150,100,50,10]
model = Sequential() #Sequential 클래스의 인스턴스
model.add(Dense(no[0], input_dim = 1)) 
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
y_prophet = [9] #<== 예측할 값
y_predict  = model.predict(y_prophet) 
print(f'{y_prophet} 의 예측값 (10,1.3,1): ',y_predict[0])
print('히든 레이어 구조 :',no) #

'''
loss :  0.00558691518381238
[9] 의 예측값 :  [9.98593    1.5621104  0.97510827]
히든 레이어 구조 : [300, 250, 200, 150, 200, 100, 50, 10, 3]

loss :  0.0292841587215662
[9] 의 예측값 (10,1.3,1):  [9.756331  1.5655575 0.8918289]
히든 레이어 구조 : [200, 150, 100, 50, 100, 150, 100, 50, 10]

loss :  0.007600471377372742
[9] 의 예측값 (10,1.3,1):  [9.949076  1.6152059 0.983785 ]
히든 레이어 구조 : [200, 150, 100, 50, 100, 150, 100, 50, 10]
'''

