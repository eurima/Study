from sklearn.datasets import fetch_covtype
from tensorflow.keras.models import Sequential , Model
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
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
         train_size = 0.8, shuffle = True, random_state = 66) 
from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
#2 모델구성
#        
deep_len = [100, 50, 30, 20, 100, 50, 30, 40, 50, 40, 30, 20, 10, 5, 4, 2]
model = Sequential() 
# model.add(LSTM(150,activation = 'relu', input_shape = (x.shape[1],1)))
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
model.add(Dense(y.shape[1], activation = 'softmax'))


#3. 컴파일, 훈련
epoch = 1
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy']) 

from tensorflow.keras.callbacks import EarlyStopping
patience_num = 50
es = EarlyStopping(monitor='val_loss', patience=patience_num, mode = 'auto', verbose=1, restore_best_weights=True)
start = time.time()

model.fit(x_train, y_train, epochs = epoch, batch_size =1000, validation_split=0.2,callbacks=[es])
end = time.time() - start


#4 평가예측
loss = model.evaluate(x_test,y_test)
print('시간 : ', round(end,2) ,'초')
print("loss : ",loss[0]) 
print("accuracy : ",loss[1])

'''
======= CONV1D ============
시간 :  14.41 초
loss :  0.875488817691803
accuracy :  0.677185595035553
'''