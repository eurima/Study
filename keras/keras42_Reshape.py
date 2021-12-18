from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Reshape, Conv1D, LSTM
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.keras.datasets import mnist
import time

(x_train, y_train),(x_test,y_test) = mnist.load_data()
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#1 데이터
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)

model = Sequential()
model.add(Conv2D(10, kernel_size=(2,2), strides=1, 
                 padding = 'same', input_shape = (x_train.shape[1],x_train.shape[2],x_train.shape[3])))#
model.add(MaxPooling2D())#14
model.add(Conv2D(5,(2,2), activation='relu')) # 13,13,5
model.add(Dropout(0.2))
model.add(Conv2D(7,(2,2), activation='relu')) # 12,12,7 아웃 
model.add(Conv2D(7,(2,2), activation='relu')) # 11,11,7 아웃 
model.add(Conv2D(10,(2,2), activation='relu')) # 10,10,10 아웃 
# model.add(Flatten()) #10*10*10 ---> (None, 1000) 
model.add(Reshape(target_shape=(100,10))) # 1000 ->100,10
model.add(Conv1D(5,2))
model.add(LSTM(15))
model.add(Dense(10, activation='softmax'))
# model.summary()
'''
'''
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
model_path = "".join([filepath,'k30_mnist2_2_',datetime,"_",filename])
es = EarlyStopping(monitor='val_loss', patience=patience_num, mode = 'auto', restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose=1, save_best_only= True, filepath = model_path)
hist = model.fit(x_train, y_train, epochs = epoch, validation_split=0.2, callbacks=[es,mcp], batch_size =1000)
end = time.time() - start
########################################################################
#4 평가예측
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
print("loss : ",loss[0])
print("accuracy : ",loss[1])
print('시간 : ', round(end,2) ,'초')
