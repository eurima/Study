from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston, load_diabetes
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
# scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

deep_len = [200,150,100,80,70,60,50,40,30,20,10,5,4,3,2,2]

model = Sequential() 
model.add(Dense(deep_len[0], input_dim = 10)) 
model.add(Dense(deep_len[1])) 
model.add(Dense(deep_len[2]))
model.add(Dense(deep_len[3],activation ='relu')) 
model.add(Dense(deep_len[4])) 
model.add(Dense(deep_len[5])) 
model.add(Dense(deep_len[6],activation ='relu')) 
model.add(Dense(deep_len[7])) 
# model.add(Dense(deep_len[8])) 
# model.add(Dense(deep_len[9])) 
# model.add(Dense(deep_len[10])) 
# model.add(Dense(deep_len[11])) 
# model.add(Dense(deep_len[12])) 
# model.add(Dense(deep_len[13])) 
# model.add(Dense(deep_len[14])) 
# model.add(Dense(deep_len[15])) 
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
model_path = "".join([filepath,'k27_diabet_',datetime,"_",filename])
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






'''

