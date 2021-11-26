from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston
import time

#1.데이터
dataset = load_boston()
x = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.8, shuffle = True, random_state = 66) 


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
epoch = 10
model.compile(loss = 'mse', optimizer = 'adam')
start = time.time()
hist = model.fit(x_train, y_train, epochs = epoch, batch_size =1,validation_split=0.2)

end = time.time() - start
print('시간 : ', round(end,2) ,'초')

#4 평가예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print("r2 : ",r2)
print(deep_len)
print("epochs :",epoch)


print('*******************')
print('hist:',hist)
print('*******************')
print('history:',hist.history)
print('*******************')
print(hist.history['loss'])
print('*******************')
print(hist.history['val_loss'])

import matplotlib.pyplot as plt
plt.figure(figsize=(9,5))
plt.plot(hist.history['loss'],marker = '.',c='red',label = 'loss')
plt.plot(hist.history['val_loss'],marker = '.',c='blue',label = 'val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')
plt.show()

'''
323/323 [==============================] - 0s 783us/step - loss: 25.7410 - val_loss: 34.9913
4/4 [==============================] - 0s 0s/step - loss: 22.9449
loss :  22.944887161254883
r2 :  0.7254836073458659
[100, 80, 60, 40, 50, 80, 70, 60, 50, 40, 30, 20, 10, 5, 4, 2]
epochs : 1000
'''

