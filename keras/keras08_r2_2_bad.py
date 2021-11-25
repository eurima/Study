import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import time

#데이터
x = np.array(range(100))
y = np.array(range(1,101))

'''
1. R2를 음수가 아닌0.5 이하로 내릴것
2. 데이터 건들지 마!
3. 레이어는 인풋 아웃풋 포함 6개이상
4. batch_size = 1
5. epochs는 100 이상
6. 히든레이어 노드는 10개이상 1000개이하
7. train 70%
'''

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.7, shuffle = True, random_state = 66) #랜덤난수 고정

# deep_len = [19, 13, 11, 17, 999]
deep_len = [random.randint(10, 1000),random.randint(10, 1000),random.randint(10, 1000),
            random.randint(10, 1000),random.randint(10, 1000),]
model = Sequential() #Sequential 클래스의 인스턴스
model.add(Dense(deep_len[0], input_dim = 1)) 
model.add(Dense(deep_len[1])) 
model.add(Dense(deep_len[2]))
model.add(Dense(deep_len[3])) 
model.add(Dense(deep_len[4])) 
model.add(Dense(deep_len[1])) 
model.add(Dense(deep_len[2]))
model.add(Dense(deep_len[3])) 
model.add(Dense(deep_len[4])) 
model.add(Dense(deep_len[1])) 
model.add(Dense(deep_len[2]))
model.add(Dense(deep_len[3])) 
model.add(Dense(deep_len[4])) 
model.add(Dense(deep_len[1])) 
model.add(Dense(deep_len[2]))
model.add(Dense(deep_len[3])) 
model.add(Dense(deep_len[4])) 
model.add(Dense(1)) 

while True:
#3. 컴파일, 훈련
    model.compile(loss = 'mse', optimizer ='adam')
    model.fit(x_train, y_train, epochs = 100, batch_size =1)

    #4 평가예측
    loss = model.evaluate(x_test,y_test)
    print("loss : ",loss)

    y_predict = model.predict(x_test)
    # plt.scatter(x, y)
    # plt.plot(x, y_predict, color = 'red')
    # plt.show()
    r2 = r2_score(y_test,y_predict)
    print("r2 : ",r2)
    print(deep_len)
    time.sleep(5)
    
    if r2 < 0.5 and r2 > 0:
        break
    
'''
r2 :  0.8752188947431137
[1000, 1000, 1000, 1000, 1000]
'''