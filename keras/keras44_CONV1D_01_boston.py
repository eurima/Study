from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
import numpy as np
import time

from sklearn.metrics import r2_score

#데이터
from sklearn.datasets import load_boston
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

dataset = load_boston()

x = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.8, shuffle = True, random_state = 66) 

scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
#[50, 20, 10, 50, 30, 15, 10, 5, 2]
deep_len = [100,80,60,40,50,80,70,60,50,40,30,20,10,5,4,2]
model = Sequential() 
model.add(Conv1D(150,2,activation = 'relu', input_shape = (x_train.shape[1],x_train.shape[2])))
model.add(Flatten())
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
epoch = 10000
model.compile(loss = 'mse', optimizer = 'adam')
from tensorflow.keras.callbacks import EarlyStopping
patience_num = 500
es = EarlyStopping(monitor='val_loss', patience=patience_num, mode = 'auto', verbose=1, restore_best_weights=True)
start = time.time()
model.fit(x_train, y_train, epochs = epoch, validation_split=0.2,callbacks=[es], batch_size =1)#batch_size =32 가 default
end = time.time() - start

#4 평가예측
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)

print("R2 : ",r2)
print("loss : ",loss)
print('시간 : ', round(end,2) ,'초')

# acc= str(loss[1]).replace(".", "_")
# model.save(f"./_save/boston_{acc}.h5")
'''
Normal
Epoch 00263: early stopping
4/4 [==============================] - 0s 988us/step - loss: 18.3059
loss :  18.305864334106445
R2 :  0.7809856267013906

MinMaxScaler
Epoch 00091: early stopping
4/4 [==============================] - 0s 1ms/step - loss: 17.2960
loss :  17.295957565307617
R2 :  0.7930683248039497


StandardScaler
Epoch 00097: early stopping
4/4 [==============================] - 0s 1ms/step - loss: 16.7891
loss :  16.78911781311035
R2 :  0.7991322361793805

RobustScaler
Epoch 00075: early stopping
4/4 [==============================] - 0s 989us/step - loss: 20.6809
loss :  20.680896759033203
R2 :  0.752570369378124


MaxAbsScaler
Epoch 00074: early stopping
4/4 [==============================] - 0s 665us/step - loss: 16.4644
loss :  16.46441078186035
R2 :  0.8030170670197926

Epoch 00103: early stopping
4/4 [==============================] - 0s 737us/step - loss: 18.6164
loss :  18.616403579711914
R2 :  0.7772702820559952

============== LSTM =======
loss :  11.208438873291016
R2 :  0.8659003999515873

========== CONV1D ========
R2 :  0.8702109313482991
loss :  10.848152160644531
시간 :  403.52 초


'''
