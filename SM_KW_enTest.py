import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score

samsung = pd.read_csv("삼성전자.csv", encoding='cp949', index_col = 0, header = 0, sep = ',',thousands=',')
kiwoom = pd.read_csv("키움증권.csv", encoding='cp949', index_col = 0, header = 0, sep = ',',thousands=',')

samsung = samsung.drop(['전일비','등락률','Unnamed: 6'], axis =1)[:800]
kiwoom = kiwoom.drop(['전일비','등락률','Unnamed: 6'], axis =1)[:800]

samsung = samsung.values
kiwoom = kiwoom.values

def split_xy(dataset, time_step, y_col, target_col):
    x,y = list(),list()
    for i in range(len(dataset)):
        x_end = i + time_step
        y_end = x_end + y_col
        
        if y_end > len(dataset):
            break
        
        temp_x = dataset[i:x_end,:]
        temp_y = dataset[x_end:y_end,target_col]
        
        x.append(temp_x)
        y.append(temp_y)
    return np.array(x),np.array(y)
time_step = 5

samsung_cx, samsung_cy = split_xy(samsung, time_step, 1, 3)#종가
kiwoom_cx, kiwoom_cy = split_xy(kiwoom, time_step, 1, 3)#종가

sc_train_x, sc_test_x, sc_train_y, sc_test_y, kc_train_x, kc_test_x, kc_train_y, kc_test_y = train_test_split(
    samsung_cx, samsung_cy, kiwoom_cx, kiwoom_cy, train_size = 0.8, shuffle = True, random_state = 66)

sc_train_shape1 = sc_train_x.shape[1]
sc_train_shape2 = sc_train_x.shape[2]
sc_test_shape1 = sc_test_x.shape[1]
sc_test_shape2 = sc_test_x.shape[2]

sc_train_x  = sc_train_x.reshape(sc_train_x.shape[0],sc_train_x.shape[1]*sc_train_x.shape[2])
sc_test_x  = sc_test_x.reshape(sc_test_x.shape[0],sc_test_x.shape[1]*sc_test_x.shape[2])
kc_train_x  = kc_train_x.reshape(kc_train_x.shape[0],kc_train_x.shape[1]*kc_train_x.shape[2])
kc_test_x  = kc_test_x.reshape(kc_test_x.shape[0],kc_test_x.shape[1]*kc_test_x.shape[2])

scaler = MaxAbsScaler()
sc_train_x = scaler.fit_transform(sc_train_x)
sc_test_x = scaler.transform(sc_test_x)
kc_train_x = scaler.fit_transform(kc_train_x)
kc_test_x = scaler.transform(kc_test_x)

sc_train_x  = sc_train_x.reshape(sc_train_x.shape[0],sc_train_shape1,sc_train_shape2)
sc_test_x  = sc_test_x.reshape(sc_test_x.shape[0],sc_test_shape1,sc_test_shape2)

kc_train_x  = kc_train_x.reshape(kc_train_x.shape[0],sc_train_shape1,sc_train_shape2)
kc_test_x  = kc_test_x.reshape(kc_test_x.shape[0],sc_test_shape1,sc_test_shape2)

# 모델구성
input1 = Input(shape = (sc_train_x.shape[1],sc_train_x.shape[2]))
dense1_1 = LSTM(15, activation='relu', )(input1)
dense1_2 = Dense(7, activation='relu', )(dense1_1)
dense1_3 = Dense(7, activation='relu', )(dense1_2)
output1 = Dense(5, activation='relu', )(dense1_3)

input2 = Input(shape = (sc_train_x.shape[1],sc_train_x.shape[2]))
dense2_1 = LSTM(15, activation='relu', )(input2)
dense2_2 = Dense(7, activation='relu', )(dense2_1)
dense2_3 = Dense(7, activation='relu', )(dense2_2)
output2 = Dense(5, activation='relu', )(dense2_3)

from tensorflow.keras.layers import Concatenate
merge1 = Concatenate()([output1,output2])

output11 = Dense(7)(merge1)
output12 = Dense(11)(output11)
output13 = Dense(11, activation = 'relu')(output12)
last_output1 = Dense(1, name='output--1')(output13)

output21 = Dense(7)(merge1)
output22 = Dense(11)(output21)
output23 = Dense(11, activation = 'relu')(output22)
last_output2 = Dense(1, name='output--2')(output23)

model = Model(inputs = [input1, input2], outputs = [last_output1,last_output2])

#3 컴파일
epochs = 10
patience =50
model.compile(loss='mse', optimizer = 'adam', metrics=['mae'])
es = EarlyStopping(monitor='val_loss', patience=patience, mode = 'auto', restore_best_weights=True)
model.fit([sc_train_x,kc_train_x], [sc_train_y,kc_train_y], epochs = epochs, validation_split=0.2, callbacks=[es],batch_size = 1)

sc_predict, kc_predict = model.predict([sc_test_x,kc_test_x])
loss = model.evaluate ([sc_test_x, kc_test_x], [sc_test_y,kc_test_y], batch_size=1)

print('Loss : ',loss)
#print("mae : ",mae1,mae2)

for i in range(5):
    print(f'삼성전자 종가 : {sc_test_y[i]} / 삼성전자 예측가 {sc_predict[i]}')
    print(f'키움증권 종가 : {kc_test_y[i]} / 키움증권 예측가 {kc_predict[i]}')

print("---------------------------------------------")
print('삼성전자 종가 예측 : ', sc_predict[-1])
print('키움증권 종가 예측 : ', kc_predict[-1])
