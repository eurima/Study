from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score

#1데이터
dataset = load_diabetes()
x = dataset.data
y = dataset.target

# print(x.shape)#442,10
# print(y.shape)#442,

# print(dataset.feature_names)
# print(dataset.DESCR)

# 기준 R2 0.62 이상

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.8, shuffle = True, random_state = 45) #랜덤난수 고정66
#[200, 180, 170, 160, 150, 140, 130, 120, 100]
deep_len = [300,150,100, 50,150,100,50,25,10]#,40,30,20,10,5,4,2]
model = Sequential() 
model.add(Dense(deep_len[0], input_dim = 10)) 
model.add(Dense(deep_len[1])) 
model.add(Dense(deep_len[2]))
model.add(Dense(deep_len[3])) 
model.add(Dense(deep_len[4])) 
model.add(Dense(deep_len[5])) 
model.add(Dense(deep_len[6])) 
model.add(Dense(deep_len[7])) 
model.add(Dense(deep_len[8])) 
# model.add(Dense(deep_len[9])) 
# model.add(Dense(deep_len[10])) 
# model.add(Dense(deep_len[11])) 
# model.add(Dense(deep_len[12])) 
# model.add(Dense(deep_len[13])) 
# model.add(Dense(deep_len[14])) 
# model.add(Dense(deep_len[15])) 
model.add(Dense(1)) 

#3. 컴파일, 훈련
epoch = 200
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = epoch, batch_size =1,verbose=2)

#4 평가예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print("r2 : ",r2)
print(deep_len)
print("epochs :",epoch)
'''
r2 :  0.46940862171453024
[500, 200, 100, 500, 300, 150, 100, 50, 20]
epochs : 100

r2 :  0.5096555698532952
[50, 20, 10, 50, 30, 15, 10, 5, 2]
epochs : 100

r2 :  0.5064730812710474
[50, 20, 10, 50, 30, 15, 10, 5, 2]
epochs : 1000

r2 :  0.5082257767857312
[50, 20, 10, 50, 30, 15, 10, 5, 2]
epochs : 500

r2 :  0.5096650556391151  ==========>
[100, 80, 70, 60, 50, 40, 30, 20, 10]
epochs : 200

r2 :  0.5149485136055756
[200, 180, 170, 160, 150, 140, 130, 120, 100]
epochs : 200

r2 :  0.568132125794226
[200, 180, 170, 160, 150, 140, 130, 120, 100]
epochs : 200 / train 0.9

r2 :  0.6185685147753804
[200, 180, 170, 160, 150, 140, 130, 120, 100] random_state = 44
epochs : 200 / train 0.9

r2 :  0.23226971022177845
[200, 180, 170, 160, 150, 140, 130, 120, 100] random_state = 45
epochs : 200 / train 0.9
'''