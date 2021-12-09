from sklearn.datasets import fetch_covtype as data_load
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from sklearn.metrics import r2_score
import numpy as np
import time

dataset  = data_load()

x = dataset.data
y = dataset.target

#print(np.unique(y)) #[1 2 3 4 5 6 7]
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.8, shuffle = True, random_state = 66) 

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
scaler = StandardScaler()

n = x_train.shape[0]
x_train_transe = scaler.fit_transform(x_train) 
print(x_train_transe.shape) #464809,54

x_train = x_train_transe.reshape(n,9,6,1) 
m = x_test.shape[0]
x_test = scaler.transform(x_test.reshape(m,-1)).reshape(m,9,6,1)

model = Sequential()
model.add(Conv2D(128, kernel_size=(4,4),padding ='same',strides=1, 
                 input_shape = (x_train.shape[1],x_train.shape[2],x_train.shape[3])))#
model.add(MaxPooling2D())
model.add(Conv2D(64,(2,2),padding ='same', activation='relu'))#<------------
# model.add(MaxPooling2D())
# model.add(Dropout(0.2))
model.add(Conv2D(32,(2,2),padding ='same', activation='relu'))
# model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(10))
# model.add(Dropout(0.2))
model.add(Dense(5))
model.add(Dense(2))
model.add(Dropout(0.5))
model.add(Dense(y.shape[1], activation = 'softmax'))

#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy']) 
########################################################################
start = time.time()
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
epoch = 10000
patience_num = 500
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")
filepath = "./_ModelCheckPoint/"
filename = '{epoch:04d}-{val_loss:4f}.hdf5' #filepath + datetime
model_path = "".join([filepath,'k35_cnn6_fetch_',datetime,"_",filename])
es = EarlyStopping(monitor='val_loss', patience=patience_num, mode = 'auto', restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose=1, save_best_only= True, filepath = model_path)
hist = model.fit(x_train, y_train, epochs = epoch, validation_split=0.2, callbacks=[es,mcp], batch_size = 50)
end = time.time() - start
print('시간 : ', round(end,2) ,'초')
########################################################################
#4 평가예측
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)

print("loss : ",loss[0]) 
print("accuracy : ",loss[1])

'''
<DNN>
시간 :  1513.77 초
3632/3632 [==============================] - 2s 570us/step - loss: 0.3160 - accuracy: 0.8784
loss :  0.3160002529621124
accuracy :  0.8783507943153381.

<CNN>
'''