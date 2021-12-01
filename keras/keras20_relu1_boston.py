
from os import scandir
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import time

# print(x.shape) 506,13
# print(y.shape) 506,

# print(dataset.feature_names)
# print(dataset.DESCR)

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

# print(np.min(x),np.max(x)) #0 , 711.0
# x = x/np.max(x)  #<======== 전체가 적용된다 나쁘지는 않지만...

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = RobustScaler()
scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#[50, 20, 10, 50, 30, 15, 10, 5, 2]
deep_len = [100,80,60,40,50,80,70,60,50,40,30,20,10,5,4,2]
model = Sequential() 
model.add(Dense(deep_len[0], input_dim =x.shape[1])) 
model.add(Dense(deep_len[1])) 
model.add(Dense(deep_len[2]))
model.add(Dense(deep_len[3])) 
model.add(Dense(deep_len[4])) 
model.add(Dense(deep_len[5] ,activation ='relu')) 
model.add(Dense(deep_len[6])) 
model.add(Dense(deep_len[7])) 
model.add(Dense(deep_len[8])) 
model.add(Dense(deep_len[9])) 
model.add(Dense(deep_len[10],activation ='relu')) 
model.add(Dense(deep_len[11])) 
model.add(Dense(deep_len[12])) 
model.add(Dense(deep_len[13])) 
model.add(Dense(deep_len[14])) 
model.add(Dense(deep_len[15])) 
model.add(Dense(1))

#3. 컴파일, 훈련
epoch = 1000
model.compile(loss = 'mse', optimizer = 'adam')
from tensorflow.keras.callbacks import EarlyStopping
patience_num = 50
es = EarlyStopping(monitor='val_loss', patience=patience_num, mode = 'auto', verbose=1, restore_best_weights=True)
start = time.time()

model.fit(x_train, y_train, epochs = epoch, validation_split=0.2,callbacks=[es], batch_size =1)#batch_size =32 가 default
end = time.time() - start


#4 평가예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print("R2 : ",r2)
print(deep_len)
print("epochs :",epoch)
'''
Normal
Epoch 00263: early stopping
4/4 [==============================] - 0s 988us/step - loss: 18.3059
loss :  18.305864334106445
R2 :  0.7809856267013906
Relu
Epoch 00080: early stopping
4/4 [==============================] - 0s 986us/step - loss: 19.4034
loss :  19.40337371826172
R2 :  0.7678548529089909

MinMaxScaler
Epoch 00091: early stopping
4/4 [==============================] - 0s 1ms/step - loss: 17.2960
loss :  17.295957565307617
R2 :  0.7930683248039497
Relu
Epoch 00070: early stopping
4/4 [==============================] - 0s 997us/step - loss: 9.7213
loss :  9.721260070800781
R2 :  0.8836932429034424

StandardScaler
Epoch 00097: early stopping
4/4 [==============================] - 0s 1ms/step - loss: 16.7891
loss :  16.78911781311035
R2 :  0.7991322361793805
Relu
Epoch 00122: early stopping
4/4 [==============================] - 0s 997us/step - loss: 7.1348
loss :  7.134823799133301
R2 :  0.9146377954919305

RobustScaler
Epoch 00075: early stopping
4/4 [==============================] - 0s 989us/step - loss: 20.6809
loss :  20.680896759033203
R2 :  0.752570369378124
Relu
Epoch 00236: early stopping
4/4 [==============================] - 0s 986us/step - loss: 7.9698
loss :  7.969822883605957
R2 :  0.9046477221544061


MaxAbsScaler
Epoch 00074: early stopping
4/4 [==============================] - 0s 665us/step - loss: 16.4644
loss :  16.46441078186035
R2 :  0.8030170670197926
Relu
Epoch 00059: early stopping
4/4 [==============================] - 0s 997us/step - loss: 13.3674
loss :  13.367378234863281
R2 :  0.8400705100856893


'''
