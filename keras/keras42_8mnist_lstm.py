from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Dropout, LSTM
import time


#1 데이터

(x_train, y_train),(x_test,y_test) = mnist.load_data()

# print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,)
x_train = x_train/255.
x_test = x_test/255.
# x_train =  x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2])
# x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2])/255.

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
patience_num = 500
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")
filepath = "./_ModelCheckPoint/"
filename = '{epoch:04d}-{val_loss:4f}.hdf5' #filepath + datetime
model_path = "".join([filepath,'k34_dnn_mnist_',datetime,"_",filename])
es = EarlyStopping(monitor='val_loss', patience=patience_num, mode = 'auto', restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose=1, save_best_only= True, filepath = model_path)
hist = model.fit(x_train, y_train, epochs = epoch, validation_split=0.2, callbacks=[es,mcp], batch_size =1000)
end = time.time() - start

########################################################################

#4 평가예측
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
print('시간 : ', round(end,2) ,'초')
print("loss : ",loss[0])
print("accuracy : ",loss[1])

'''
<DNN 결과>>
시간 :  130.72 초
313/313 [==============================] - 0s 863us/step - loss: 0.1483 - accuracy: 0.9650
loss :  0.1483190357685089
accuracy :  0.9649999737739563

<CNN 결과> >>>>>>>>>>>>>>>>>>>>
시간 :  4441.69 초
313/313 [==============================] - 1s 2ms/step - loss: 0.0834 - accuracy: 0.9765
loss :  0.08340967446565628
accuracy :  0.9764999747276306

============== LSTM ====================
시간 :  19076.78 초
313/313 [==============================] - 2s 7ms/step - loss: 0.1565 - accuracy: 0.9735
loss :  0.15645381808280945
accuracy :  0.9735000133514404


'''