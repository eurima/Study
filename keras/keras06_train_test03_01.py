import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import random

#1. 데이터
x = np.array(range(100))
y = np.array(range(1,101))

#훈련과 테스트를 7:3으로 주고 섞어서 완성
#내가만든 막 코딩

test_rate = 0.3
train_rate = 1 - test_rate
test_num = len(x)*test_rate
train_num = len(x)*train_rate

test_index_num =set()
while True:
    i = random.randint(0,len(x)-1)
    test_index_num.add(i)
    if len(test_index_num) >= test_num:
        break
    
# print("test_index_num : ",test_index_num)

x_test = np.array([])
y_test = np.array([])
x_train = np.array([])
y_train = np.array([])

for i in test_index_num:
    x_test = np.append(x_test, np.array(x[i]))
    y_test = np.append(y_test, np.array(y[i]))


index = list(test_index_num)
x_train = np.delete(x, index)
y_train = np.delete(y, index)


#2. 모델
deep_len = [250,200,150,100,200,150,100,50,10]
model = Sequential() #Sequential 클래스의 인스턴스
model.add(Dense(deep_len[0], input_dim = 1)) 
model.add(Dense(deep_len[1])) 
model.add(Dense(deep_len[2]))
model.add(Dense(deep_len[3])) 
model.add(Dense(deep_len[4])) 
model.add(Dense(deep_len[5])) 
model.add(Dense(deep_len[6])) 
model.add(Dense(deep_len[7])) 
model.add(Dense(deep_len[8])) 
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

print("x_test : ", x_test)
print("y_test : ", y_test)

print("x_train : ",x_train)
print("y_train : ",y_train)

print(len(x_test))
print(len(x_train))

'''
Loss =  0.0013888335088267922
[11]의 예측값 : [[11.980751]]
'''