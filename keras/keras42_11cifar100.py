from tensorflow.keras.datasets import cifar100

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Dropout,LSTM
import numpy as np
import time

#1 데이터

(x_train, y_train),(x_test,y_test) = cifar100.load_data()

dim = x_train.shape[1]*x_train.shape[2]*x_train.shape[3]
out_node = len(np.unique(y_train))

x_train = x_train/255.
x_test = x_test/255.

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[3],x_train.shape[2])
x_train = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[3],x_test.shape[2])

deep_len = [100,80,60,40,50,80,70,60,50,40,30,20,10,5,4,2]
model = Sequential()
model.add(LSTM(150,activation = 'relu', input_shape = (x_train.shape[1],x_train.shape[2])))
model.add(Dense(128, input_dim = dim))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(out_node, activation='softmax'))


#3. 컴파일, 훈련

opt="adam"
model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics=['accuracy']) # metrics=['accuracy'] 영향을 미치지 않는다
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
model_path = "".join([filepath,'k35_cfar10_',datetime,"_",filename])
es = EarlyStopping(monitor='val_loss', patience=patience_num, mode = 'auto', restore_best_weights=True)
# mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose=1, save_best_only= True, filepath = model_path)
hist = model.fit(x_train, y_train, epochs = epoch, validation_split=0.2, callbacks=[es], batch_size =500)
end = time.time() - start
print('시간 : ', round(end,2) ,'초')
########################################################################

#4 평가예측
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
print("loss : ",loss[0])
print("accuracy : ",loss[1])
'''
<CNN>>>>
시간 :  5163.34 초
313/313 [==============================] - 4s 14ms/step - loss: 2.6845 - accuracy: 0.3223
loss :  2.6844635009765625
accuracy :  0.322299987077713

<DNN>
시간 :  2954.8 초
313/313 [==============================] - 1s 4ms/step - loss: 4.6052 - accuracy: 0.0100
loss :  4.60524845123291
accuracy :  0.009999999776482582

=========== LSTM ==========================
시간 :  183.33 초
loss :  0.9735575318336487
accuracy :  0.622772216796875

'''