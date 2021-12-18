from tensorflow.keras.datasets import fashion_mnist

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Dropout, LSTM
import numpy as np
import time

#1 데이터

(x_train, y_train),(x_test,y_test) = fashion_mnist.load_data()

dim = x_train.shape[1]*x_train.shape[1]
out_node = len(np.unique(y_train))

x_train = x_train/255.
x_test = x_test/255.

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


deep_len = [100,80,60,40,50,80,70,60,50,40,30,20,10,5,4,2]
model = Sequential()

model.add(LSTM(150,activation = 'relu', input_shape = (x_train.shape[1],x_train.shape[2])))
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
model_path = "".join([filepath,'k35_fashion_',datetime,"_",filename])
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
<CNN>>>
시간 :  1582.41 초
313/313 [==============================] - 1s 2ms/step - loss: 0.3688 - accuracy: 0.8705
loss :  0.3687971234321594
accuracy :  0.8705000281333923

<DNN>
시간 :  179.13 초
313/313 [==============================] - 0s 976us/step - loss: 0.4245 - accuracy: 0.8636
loss :  0.4245057702064514
accuracy :  0.8636000156402588


========== LSTM ===============
시간 :  12014.77 초
313/313 [==============================] - 11s 17ms/step - loss: 0.4271 - accuracy: 0.8739
loss :  0.4271278381347656
accuracy :  0.8738999962806702


'''


