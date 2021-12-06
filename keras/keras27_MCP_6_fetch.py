from sklearn.datasets import fetch_covtype
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import time

#1 데이터
dataset  = fetch_covtype()
# print(dataset)
# print(dataset.DESCR) 

x = dataset.data
y = dataset.target #===== sklearn에서만 제공!!
# print(x.shape, y.shape) 
# print(np.unique(y)) #---->  배열의 고유값을 찾아준다 (라벨값이 어떤것이 있는가) len(np.unique(y))

from tensorflow.keras.utils import to_categorical
y = to_categorical(y) 

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.8, shuffle = True, random_state = 66) #

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
#2 모델구성
#        
deep_len = [100, 50, 30, 20, 100, 50, 30, 40, 50, 40, 30, 20, 10, 5, 4, 2]
model = Sequential()
model.add(Dense(deep_len[0], activation = 'linear', input_dim =x.shape[1]))
model.add(Dense(deep_len[1], )) 
model.add(Dense(deep_len[2]))
model.add(Dense(deep_len[3])) 
model.add(Dense(deep_len[4])) 
model.add(Dense(deep_len[5], activation = 'relu')) 
model.add(Dense(deep_len[6])) 
model.add(Dense(deep_len[7])) 
model.add(Dense(deep_len[8])) 
model.add(Dense(deep_len[9])) 
model.add(Dense(deep_len[10], activation = 'relu')) 
model.add(Dense(deep_len[11])) 
model.add(Dense(deep_len[12])) 
model.add(Dense(deep_len[13])) 
model.add(Dense(deep_len[14])) 
model.add(Dense(deep_len[15])) 
model.add(Dense(y.shape[1], activation = 'softmax'))


#3. 컴파일, 훈련
epoch = 10000
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy']) 
########################################################################
# model.compile(loss = 'mse', optimizer = 'adam')
start = time.time()
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
epoch = 10000
patience_num = 500
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")
filepath = "./_ModelCheckPoint/"
filename = '{epoch:04d}-{val_loss:4f}.hdf5' #filepath + datetime
model_path = "".join([filepath,'k27_fetch_',datetime,"_",filename])
es = EarlyStopping(monitor='val_loss', patience=patience_num, mode = 'auto', restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose=1, save_best_only= True, filepath = model_path)
model.fit(x_train, y_train, epochs = epoch, validation_split=0.2,callbacks=[es,mcp], batch_size =1)
end = time.time() - start
print('시간 : ', round(end,2) ,'초')
########################################################################

#4 평가예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss[0]) 
print("accuracy : ",loss[1])

y_predict = model.predict(x_test)
print("epochs :",epoch)
# result = y_predict[:7]
# print(result)
# print(y_test[:7])
'''
<<기존 성과 우수>>
======StandardScaler
Epoch 00095: early stopping
시간 :  459.15 초
3632/3632 [==============================] - 2s 496us/step - loss: 0.6618 - accuracy: 0.7218
loss :  0.6617664694786072
accuracy :  0.7217885851860046
<Relu>
Epoch 00283: early stopping
시간 :  1513.77 초
3632/3632 [==============================] - 2s 570us/step - loss: 0.3160 - accuracy: 0.8784
loss :  0.3160002529621124
accuracy :  0.8783507943153381.
<<ModelCheckpoint >>

Epoch 00020: val_loss did not improve from 0.63842
Epoch 21/10000
 97828/371847 [======>.......................] - ETA: 3:27 - loss: 0.6660 - accuracy: 0.7270    


'''
