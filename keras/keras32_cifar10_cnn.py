from tensorflow.keras.datasets import cifar10
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from tensorflow.python.keras.layers.core import Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time

(x_train, y_train),(x_test,y_test) = cifar10.load_data()

# print(x_train.shape)
# print(y_train.shape)
# print(np.unique(y_train,return_counts=True))
'''
(50000, 32, 32, 3)
(50000, 1)
(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],
      dtype=int64))
'''
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
'''
scaler = StandardScaler()
n = x_train.shape[0]# 이미지갯수 50000
x_train_reshape = x_train.reshape(n,-1) #----> (50000,32,32,3) --> (50000, 32*32*3 ) 0~255
#x_train_transe = scaler.fit_transform(x_train_reshape) # 0 ~ 255 -> 0.0 ~ 1.0
x_train_transe = x_train_reshape /255. #---- scale 안쓰고 scale 하기
x_train = x_train_transe.reshape(x_train.shape) #--->(50000,32,32,3) 0.0 ~ 1.0

m = x_test.shape[0]
x_test = scaler.transform(x_test.reshape(m,-1)).reshape(x_test.shape)
'''
x_train = x_train/255.
x_test = x_test/255.


model = Sequential()
model.add(Conv2D(64, kernel_size=(4,4),padding ='same',strides=1, input_shape = (32,32,3)))
model.add(MaxPool2D(2))
model.add(Conv2D(32,(4,4),padding ='same', activation='relu'))
model.add(MaxPool2D(2))
model.add(Dropout(0.2))
model.add(Conv2D(8,(2,2),padding ='same', activation='relu'))
model.add(MaxPool2D(2))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dense(25))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

#model.summary()#3,153
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
model_path = "".join([filepath,'k32_cifar10_',datetime,"_",filename])
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
전처리 전
시간 :  1355.58 초
313/313 [==============================] - 1s 3ms/step - loss: 1.5798 - accuracy: 0.3749
loss :  1.5797550678253174
accuracy :  0.3749000132083893

전처리 후
시간 :  4565.11 초
313/313 [==============================] - 2s 5ms/step - loss: 0.9039 - accuracy: 0.6799
loss :  0.9039157032966614
accuracy :  0.6798999905586243
'''

