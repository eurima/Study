import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array(range(100))
y = np.array(range(1,101))

#훈련과 테스트를 7:3으로 주고 섞어서 완성
#남들이 만든거 쓰는것도 실력이다!! 우리는 Developer ~!

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.7, shuffle = True, random_state = 66) #랜덤난수 고정

# print(x_test)
# print(y_test)

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
res =[100]
result = model.predict(res)
print(f'{res}의 예측값 :',result)

'''
Loss =  0.008211896754801273
[100]의 예측값 : [[101.02252]]

'''

