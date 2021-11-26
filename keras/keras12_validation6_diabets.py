from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score

#1데이터
dataset = load_diabetes()
x = dataset.data
y = dataset.target

print(len(x))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.7, shuffle = True, random_state = 45) 

# [50, 20, 10, 50, 30, 15, 10, 5, 2]
# deep_len = [100,80,60,40,50,80,70,60,50,40,30,20,10,5,4,2]
# model = Sequential() 
# model.add(Dense(deep_len[0], input_dim = 10)) 
# model.add(Dense(deep_len[1])) 
# model.add(Dense(deep_len[2]))
# model.add(Dense(deep_len[3])) 
# model.add(Dense(deep_len[4])) 
# model.add(Dense(deep_len[5])) 
# model.add(Dense(deep_len[6])) 
# model.add(Dense(deep_len[7])) 
# model.add(Dense(deep_len[8])) 
# model.add(Dense(deep_len[9])) 
# model.add(Dense(deep_len[10])) 
# model.add(Dense(deep_len[11])) 
# model.add(Dense(deep_len[12])) 
# model.add(Dense(deep_len[13])) 
# model.add(Dense(deep_len[14])) 
# model.add(Dense(deep_len[15])) 
# model.add(Dense(1)) 


deep_len = [500,400,300,200,150,100,50,25,10]#,40,30,20,10,5,4,2]
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
model.add(Dense(1)) 

#3. 컴파일, 훈련
epoch = 500
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = epoch, batch_size =1,verbose=2,validation_split=0.426)

#4 평가예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print("R2 : ",r2)
print(deep_len)
print("epochs :",epoch)


'''
264/264 - 0s - loss: 3046.4243 - val_loss: 3413.2910
3/3 [==============================] - 0s 0s/step - loss: 3964.9304
loss :  3964.930419921875
r2 :  0.38907458266545913
[100, 80, 60, 40, 50, 80, 70, 60, 50, 40, 30, 20, 10, 5, 4, 2]
epochs : 200
train_size = 0.8
validation_split=0.25

3/3 [==============================] - 0s 0s/step - loss: 3390.3147
loss :  3390.314697265625
R2 :  0.47761272269368193
[100, 80, 60, 40, 50, 80, 70, 60, 50, 40, 30, 20, 10, 5, 4, 2]
epochs : 1000
train_size = 0.8
validation_split=0.25

2/2 [==============================] - 0s 0s/step - loss: 3496.4211
loss :  3496.421142578125
R2 :  0.560678210718125
[100, 80, 60, 40, 50, 80, 70, 60, 50, 40, 30, 20, 10, 5, 4, 2]
epochs : 300
train_size = 0.9
validation_split=0.11

5/5 [==============================] - 0s 0s/step - loss: 3364.4343
loss :  3364.434326171875
R2 :  0.45999446249816456
[100, 80, 60, 40, 50, 80, 70, 60, 50, 40, 30, 20, 10, 5, 4, 2]
epochs : 300
train_size = 0.7
validation_split=0.426

5/5 [==============================] - 0s 0s/step - loss: 3186.2703
loss :  3186.270263671875
R2 :  0.4885905503127558
[100, 80, 60, 40, 50, 80, 70, 60, 50, 40, 30, 20, 10, 5, 4, 2]
epochs : 1000

5/5 [==============================] - 0s 1ms/step - loss: 2633.6406 
loss :  2633.640625
R2 :  0.5062352645718222 <===============
[500, 400, 300, 200, 150, 100, 50, 25, 10]
epochs : 200

177/177 - 0s - loss: 3275.4417 - val_loss: 3451.8955
5/5 [==============================] - 0s 0s/step - loss: 2594.5344
loss :  2594.534423828125
R2 :  0.5135670752469419
[500, 400, 300, 200, 150, 100, 50, 25, 10]
epochs : 500
'''