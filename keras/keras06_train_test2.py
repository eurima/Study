import numpy as np
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
#test와 train을 8:2 으로 분리하시오
div_num1 = 8 
div_num2 = 10-div_num1+1

x_train = x[:div_num1]
y_train = y[:div_num1]

# print(x_train) #[1,2,3,4,5,6,7,8]

x_test = x[:-(div_num2):-1]
y_test = y[:-(div_num2):-1]

# print(x_test) #[10,9]

#2. 모델
no = [250,200,150,100,200,150,100,50,10]
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

print("x_train",x_train)
print("y_train",y_train)

print("x_test",x_test) 
print("y_test",y_test) 


'''
Loss =  1.1095835361629725e-10
[11]의 예측값 : [[11.000014]]

Loss =  6.965176726225764e-05   <--------------------
[11]의 예측값 : [[10.990579]]
x_train = [1 2 3 4 5 6 7 8]
x_test = [10  9]
y_train = [1 2 3 4 5 6 7 8]
y_test = [10  9]

'''

