import numpy as np
from sklearn.metrics import r2_score
import time
#1. 데이터
x1 = np.array([range(100), range(301, 401)]) #ex) 삼성전자의 저가, 고가
x2 = np.array([range(101, 201), range(411, 511), range(100, 200)]) #미국선물의 시가, 고가, 종가
x1 = np.transpose(x1) #(100,2)으로 변환하기 위함
x2 = np.transpose(x2)

y = np.array(range(1001, 1101)) #ex)삼성전자의 종가
# print(x1.shape, x2.shape, y.shape) #(100, 2) (100, 3) (100,)

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, train_size = 0.8, shuffle = True, random_state = 66)

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
merge1 = Concatenate()[output1,output2]
merge2 = Dense(10)(merge1)
merge3 = Dense(7)(merge2)
last_output = Dense(1)(merge3)

from tensorflow.keras.models import Sequential , Model
model = Model(inputs = [input1, input2], outputs = last_output)
# 

#3. 컴파일, 훈련
epoch = 100000
model.compile(loss = 'mae', optimizer = 'adam')
from tensorflow.keras.callbacks import EarlyStopping
patience_num = 500
es = EarlyStopping(monitor='val_loss', patience=patience_num, mode = 'auto', verbose=1, restore_best_weights=True)
# start = time.time()
model.fit([x1_train,x2_train], y_train, epochs = epoch, validation_split=0.2,callbacks=[es], batch_size =1)

#4 평가예측
loss = model.evaluate([x1_test,x2_test],y_test)
y_predict = model.predict([x1_test,x2_test])
r2 = r2_score(y_test,y_predict)

print("R2 : ",r2)
print("loss : ",loss)

# print('시간 : ', round(end,2) ,'초')

'''
R2 :  0.9993876532333354
loss :  0.5941131711006165

R2 :  0.9999680336828494
loss :  0.11450805515050888
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_2 (InputLayer)            [(None, 3)]          0
__________________________________________________________________________________________________
input_1 (InputLayer)            [(None, 2)]          0
__________________________________________________________________________________________________
dense11 (Dense)                 (None, 10)  3*10+10         40          input_2[0][0]
__________________________________________________________________________________________________
dense1 (Dense)                  (None, 5)     2*5+5       15          input_1[0][0]
__________________________________________________________________________________________________
dense12 (Dense)                 (None, 10)   10*10 +10        110         dense11[0][0]
__________________________________________________________________________________________________
dense2 (Dense)                  (None, 7)    5*7+7        42          dense1[0][0]
__________________________________________________________________________________________________
dense13 (Dense)                 (None, 10)   10*10+10        110         dense12[0][0]
__________________________________________________________________________________________________
dense3 (Dense)                  (None, 7)   7*7+7         56          dense2[0][0]
__________________________________________________________________________________________________
dense14 (Dense)                 (None, 10)  10*10+10         110         dense13[0][0]
__________________________________________________________________________________________________
output1 (Dense)                 (None, 7)  7*7+7          56          dense3[0][0]
__________________________________________________________________________________________________
output2 (Dense)                 (None, 5)  10*5+5          55          dense14[0][0]
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 12)   7+5        0           output1[0][0]
                                                                 output2[0][0]
__________________________________________________________________________________________________
dense (Dense)                   (None, 10)     12*10+10      130         concatenate[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 7)    7*10+7        77          dense[0][0]
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 1)    7*1+1        8           dense_1[0][0]
==================================================================================================
Total params: 809
Trainable params: 809
'''
# model.summary()







