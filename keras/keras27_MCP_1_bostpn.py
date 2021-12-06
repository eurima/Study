
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
scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = MaxAbsScaler()
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

########################################################################
model.compile(loss = 'mse', optimizer = 'adam')
start = time.time()
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
epoch = 10000
patience_num = 500
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")
filepath = "./_ModelCheckPoint/"
filename = '{epoch:04d}-{val_loss:4f}.hdf5' #filepath + datetime
model_path = "".join([filepath,'k27_boston_',datetime,"_",filename])
es = EarlyStopping(monitor='val_loss', patience=patience_num, mode = 'auto', restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose=1, save_best_only= True, filepath = model_path)
model.fit(x_train, y_train, epochs = epoch, validation_split=0.2,callbacks=[es,mcp], batch_size =1)
end = time.time() - start
########################################################################
#4 평가예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print("R2 : ",r2)
print(deep_len)
print("epochs :",epoch)
'''
<<기존 성과>>
StandardScaler
Relu
Epoch 00122: early stopping
4/4 [==============================] - 0s 997us/step - loss: 7.1348
loss :  7.134823799133301
R2 :  0.9146377954919305

<<ModelCheckpoint 우수>>
Epoch 01423: val_loss did not improve from 6.13479
4/4 [==============================] - 0s 0s/step - loss: 5.5674
loss :  5.567386627197266
R2 :  0.9333908703456402



'''
