import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from tensorflow.python.keras.layers.core import Dropout
import time

(x_train, y_train),(x_test,y_test) = mnist.load_data()
out_node = len(np.unique(y_train))

x_train = x_train/255.
x_test = x_test/255.

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[3],x_train.shape[2])
# x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[3],x_test.shape[2])

deep_len = [100,80,60,40,50,80,70,60,50,40,30,20,10,5,4,2]
model = Sequential()
model.add(Conv1D(150,2,activation = 'relu', input_shape = (x_train.shape[1],x_train.shape[2])))
model.add(Flatten())
# model.add(LSTM(150,activation = 'relu', input_shape = (x_train.shape[1],x_train.shape[2])))
model.add(Dense(128))
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
Epoch 00258: val_loss did not improve from 0.08222
시간 :  1530.12 초
375/375 [==============================] - 1s 2ms/step - loss: 0.0766 - accuracy: 0.9768
loss :  0.07659225165843964
accuracy :  0.9768333435058594

<Conv1D>
시간 :  1318.41 초
loss :  0.12515005469322205
accuracy :  0.9782999753952026



'''

 