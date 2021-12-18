import numpy as np
from sklearn.metrics import r2_score
import time
#1. 데이터
x1 = np.array([range(100), range(301, 401)]) #ex) 삼성전자의 저가, 고가
x2 = np.array([range(101, 201), range(411, 511), range(100, 200)]) #미국선물의 시가, 고가, 종가
x1 = np.transpose(x1) #(100,2)으로 변환하기 위함
x2 = np.transpose(x2)

y1 = np.array(range(1001, 1101)) 
y2 = np.array(range(101, 201)) 

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y1_train, y1_test , y2_train, y2_test = train_test_split(
    x1, x2, y1, y2, train_size = 0.8, shuffle = True, random_state = 66)

print(x1_train.shape,x1_test.shape)
print(x2_train.shape,x2_test.shape)
print(y1_train.shape,y1_test.shape)
print(y1_train.shape,y2_test.shape)

#2 모델구성
from tensorflow.keras.layers import Dense, Input
#2-1
input1 = Input(shape=(x1_train.shape[1],))
dense1 = Dense(5, activation='relu', name='dense1')(input1)
dense2 = Dense(7, activation='relu', name='dense2')(dense1)
dense3 = Dense(7, activation='relu', name='dense3')(dense2)
output1 = Dense(7, activation='relu', name='output1')(dense3)

#2-2
input2 = Input(shape=(x2_train.shape[1],))
dense11 = Dense(10, activation='relu', name='dense11')(input2)
dense12 = Dense(10, activation='relu', name='dense12')(dense11)
dense13 = Dense(10, activation='relu', name='dense13')(dense12)
dense14 = Dense(10, activation='relu', name='dense14')(dense13)
output2 = Dense(5, activation='relu', name='output2')(dense14)

from tensorflow.keras.layers import concatenate, Concatenate
# merge1 = concatenate([output1,output2])
merge1 = Concatenate()([output1,output2])

output21 = Dense(7)(merge1)
output22 = Dense(11)(output21)
output23 = Dense(11, activation = 'relu')(output22)
last_output1 = Dense(1, name='output--1')(output23)

output31 = Dense(7)(merge1)
output32 = Dense(21)(output31)
output33 = Dense(21)(output32)
output34 = Dense(21, activation = 'relu')(output33)
last_output2 = Dense(1, name='output--2')(output34)

from tensorflow.keras.models import Sequential , Model
model = Model(inputs = [input1, input2], outputs = [last_output1,last_output2])

#3. 컴파일, 훈련
epoch = 100
model.compile(loss = 'mae', optimizer = 'adam',metrics=['mae'])
from tensorflow.keras.callbacks import EarlyStopping
patience_num = 5
es = EarlyStopping(monitor='val_loss', patience=patience_num, mode = 'auto', verbose=1, restore_best_weights=True)
# start = time.time()
model.fit([x1_train,x2_train], [y1_train,y2_train], epochs = epoch, validation_split=0.2,callbacks=[es], batch_size =1)

#4 평가예측
result = model.evaluate([x1_test,x2_test],[y1_test,y2_test])
# y1_predict = model.predict(x1_test)
# r2_1 = r2_score(y1_test,y1_predict)
# y2_predict = model.predict(x2_test)
# r2_2 = r2_score(y2_test,y2_predict)

# print("R2 : ",r2_1)
# print("R2 : ",r2_2)
print("loss : ",result[0])








