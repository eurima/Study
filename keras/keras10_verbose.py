import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import time


#데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.7, shuffle = True, random_state = 66) #랜덤난수 고정

#2 모델
start_time = time.time()

deep_len = [5,5,5,5,5]
model = Sequential() #Sequential 클래스의 인스턴스
model.add(Dense(deep_len[0], input_dim = 1)) 
model.add(Dense(deep_len[1])) 
model.add(Dense(deep_len[2]))
model.add(Dense(deep_len[3])) 
model.add(Dense(deep_len[4])) 

model.add(Dense(1)) 

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y, epochs = 1000, batch_size =1, verbose=1)
'''
verbos - 
0  없다
1  모두 보여짐
2  loss
3~ epoch만
'''
'''
#4 평가예측
loss = model.evaluate(x,y)
print("loss : ",loss)

y_predict = model.predict(x)
r2 = r2_score(y,y_predict)
print("r2 : ",r2)
print(deep_len)
'''
end_time = time.time()
print("실행시간 : ", end_time - start_time) #0:2.63570 1:3.65172