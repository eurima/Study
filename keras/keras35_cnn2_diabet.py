from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes
import time
#1.데이터
dataset = load_diabetes()
x = dataset.data
y = dataset.target
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.8, shuffle = True, random_state = 66) 

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
scaler = MinMaxScaler()


n = x_train.shape[0]# 
# x_train_reshape = x_train.reshape(n,-1) #----> (50000,32,32,3) --> (50000, 32*32*3 ) 0~255
x_train_transe = scaler.fit_transform(x_train) 
print(x_train_transe.shape) #353,10

x_train = x_train_transe.reshape(n,2,5,1) 
m = x_test.shape[0]
x_test = scaler.transform(x_test.reshape(m,-1)).reshape(m,2,5,1)

model = Sequential()
model.add(Conv2D(128, kernel_size=(4,4),padding ='same',strides=1, input_shape = (2,5,1)))#
model.add(MaxPooling2D())
model.add(Conv2D(64,(2,2),padding ='same', activation='relu'))#<------------
# model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Conv2D(32,(2,2),padding ='same', activation='relu'))
# model.add(MaxPooling2D())
model.add(Flatten())
# print("==========================================★")
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dense(20))
model.add(Dropout(0.5))
model.add(Dense(1))

#model.summary()#3,153
#3. 컴파일, 훈련

opt="adam"
model.compile(loss = 'mse', optimizer = opt) # metrics=['accuracy'] 영향을 미치지 않는다
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
model_path = "".join([filepath,'k35_cnn_boston_',datetime,"_",filename])
es = EarlyStopping(monitor='val_loss', patience=patience_num, mode = 'auto', restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose=1, save_best_only= True, filepath = model_path)
hist = model.fit(x_train, y_train, epochs = epoch, validation_split=0.2, callbacks=[es,mcp], batch_size = 50)
end = time.time() - start
print('시간 : ', round(end,2) ,'초')
########################################################################
#4 평가예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print("R2 : ",r2)
print("epochs :",epoch)
'''
<<기존 성과 우수>>
MinMaxScaler
Epoch 00059: early stopping
시간 :  12.34 초
3/3 [==============================] - 0s 999us/step - loss: 3194.6562
loss :  3194.65625
R2 :  0.5077602072769352
Relu
Epoch 00057: early stopping
시간 :  12.23 초
3/3 [==============================] - 0s 1ms/step - loss: 3255.7739
loss :  3255.77392578125
R2 :  0.49834307650366017

<<ModelCheckpoint>>
Epoch 00683: val_loss did not improve from 2664.36499
3/3 [==============================] - 0s 997us/step - loss: 3955.6995
loss :  3955.699462890625
R2 :  0.390497062159188

<<CNN>>
시간 :  18.19 초
3/3 [==============================] - 0s 998us/step - loss: 3936.3647
loss :  3936.36474609375
R2 :  0.3934760247039304
'''