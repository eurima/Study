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
         train_size = 0.9, shuffle = True, random_state = 66) 

[50, 20, 10, 50, 30, 15, 10, 5, 2]
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
model.fit(x_train, y_train, epochs = epoch, batch_size =1,validation_split=0.1)

#4 평가예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print("r2 : ",r2)
print(deep_len)
print("epochs :",epoch)

'''
316/316 [==============================] - 0s 840us/step - loss: 27.4460 - val_loss: 30.6392
3/3 [==============================] - 0s 0s/step - loss: 19.2068
loss :  19.206785202026367
r2 :  0.7837500797399244
[100, 80, 60, 40, 50, 80, 70, 60, 50, 40, 30, 20, 10, 5, 4, 2]
epochs : 1000

validation_split=0.23
train_size = 0.8125

Epoch 1000/1000
409/409 [==============================] - 0s 756us/step - loss: 26.3723 - val_loss: 22.5501
2/2 [==============================] - 0s 0s/step - loss: 21.3376
loss :  21.33759117126465
r2 :  0.7770126366213838
[100, 80, 60, 40, 50, 80, 70, 60, 50, 40, 30, 20, 10, 5, 4, 2]
epochs : 1000

validation_split=0.1
train_size = 0.9

'''
