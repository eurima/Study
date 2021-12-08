import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.python.keras.layers.core import Dropout
import time

(x_train, y_train),(x_test,y_test) = fashion_mnist.load_data()

# print(x_train.shape) #(60000, 28, 28)
# print(y_train.shape) #(60000, 28, 28)

# print(np.unique(y_train,return_counts=True))
'''
(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000],
      dtype=int64)).
'''

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#1 데이터
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)


model = Sequential()
model.add(Conv2D(128, kernel_size=(2,2), input_shape = (28,28,1)))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(32,(2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(128))
model.add(Dropout(0.2))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


#3. 컴파일, 훈련

opt="adam"
model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics=['accuracy']) # metrics=['accuracy'] 영향을 미치지 않는다
########################################################################
# model.compile(loss = 'mse', optimizer = 'adam')
start = time.time()
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
epoch = 10000
patience_num = 50
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")
filepath = "./_ModelCheckPoint/"
filename = '{epoch:04d}-{val_loss:4f}.hdf5' #filepath + datetime
model_path = "".join([filepath,'k31_mnist2_2_',datetime,"_",filename])
es = EarlyStopping(monitor='val_loss', patience=patience_num, mode = 'auto', restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose=1, save_best_only= True, filepath = model_path)
hist = model.fit(x_train, y_train, epochs = epoch, validation_split=0.2, callbacks=[es,mcp], batch_size =500)
end = time.time() - start
print('시간 : ', round(end,2) ,'초')
########################################################################

#4 평가예측
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
print("loss : ",loss[0])
print("accuracy : ",loss[1])

'''
시간 :  1582.41 초
313/313 [==============================] - 1s 2ms/step - loss: 0.3688 - accuracy: 0.8705
loss :  0.3687971234321594
accuracy :  0.8705000281333923

Epoch 00067: val_loss did not improve from 0.31537
Epoch 68/10000
40/96 [===========>..................] - ETA: 43s - loss: 0.1408 - accuracy: 0.9515 
'''


