from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np



# print(x.shape) 506,13
# print(y.shape) 506,

# print(dataset.feature_names)
# print(dataset.DESCR)

from sklearn.metrics import r2_score

#데이터
from sklearn.datasets import load_boston
from tensorflow.python.keras.callbacks import EarlyStopping
dataset = load_boston()
x = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.7, shuffle = True, random_state = 66) #랜덤난수 고정
#[50, 20, 10, 50, 30, 15, 10, 5, 2]
deep_len = [100,80,60,40,50,80,70,60,50,40,30,20,10,5,4,2]
model = Sequential() 
model.add(Dense(deep_len[0], input_dim = 13)) 
model.add(Dense(deep_len[1])) 
model.add(Dense(deep_len[2]))
model.add(Dense(deep_len[3])) 
model.add(Dense(deep_len[4])) 
model.add(Dense(deep_len[5])) 
model.add(Dense(deep_len[6])) 
model.add(Dense(deep_len[7])) 
model.add(Dense(deep_len[8])) 
model.add(Dense(deep_len[9])) 
model.add(Dense(deep_len[10])) 
model.add(Dense(deep_len[11])) 
model.add(Dense(deep_len[12])) 
model.add(Dense(deep_len[13])) 
model.add(Dense(deep_len[14])) 
model.add(Dense(deep_len[15])) 
model.add(Dense(1)) 

#3. 컴파일, 훈련
epoch = 1000
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = epoch, batch_size =1)

#4 평가예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print("r2 : ",r2)
print(deep_len)
print("epochs :",epoch)

'''
loss :  20.52109718322754
r2 :  0.7516119657162676
[500, 200, 100, 200, 10]

loss :  18.046527862548828
r2 :  0.7815642570169652
[50, 20, 10, 50, 30, 15, 10, 5, 2]
epochs = 2000

loss :  18.024595260620117
r2 :  0.7818297266155326
[50, 20, 10, 50, 30, 15, 10, 5, 2]
epochs : 4000

loss :  17.256607055664062   
r2 :  0.7911254681168102      <---------------------------
[100, 80, 60, 40, 50, 80, 70, 60, 50, 40, 30, 20, 10, 5, 4, 2]
epochs : 1000

'''
