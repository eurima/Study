import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import os

'''
앙상블 : 103 ~ 142 라인

LSTM : 105, 111 라인

금요일까지의 데이터로 월, 화 데이터 예측 : 145번 라인 ~
'''

################### 폴더 관리 ###########################################
file_name = os.path.abspath(__file__)
filepath = "./_ModelCheckPoint/"
dir_name = filepath + file_name.split("\\")[-1].split('.')[0]
os.makedirs(dir_name, exist_ok=True)

filepath = dir_name  # "./_ModelCheckPoint/"
filename = '{epoch:04d}-{val_loss:4f}.hdf5'
model_path = "".join([file_name.split("\\")[-1].split('.')[0], "_", filename])
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True,
                      filepath=filepath + "\\" + model_path)
############################
samsung = pd.read_csv("삼성전자.csv", encoding='cp949', index_col=0, header=0, sep=',', thousands=',')[:880]
kiwoom = pd.read_csv("키움증권.csv", encoding='cp949', index_col=0, header=0, sep=',', thousands=',')[:880]

samsung = samsung.drop(['전일비', '등락률', 'Unnamed: 6', '금액(백만)', '신용비', '외국계', '프로그램', '외인비'], axis=1)
kiwoom = kiwoom.drop(['전일비', '등락률', 'Unnamed: 6', '금액(백만)', '신용비', '외국계', '프로그램', '외인비'], axis=1)

samsung = samsung.sort_values(['일자'], ascending=True)
samsung['MA20'] = samsung['종가'].rolling(window=20).mean()
samsung['MA60'] = samsung['종가'].rolling(window=60).mean()
samsung['stddev'] = samsung['종가'].rolling(window=20).std()
samsung['Boll_UP'] = samsung['MA20'] + samsung['stddev'] * 2
samsung['Boll_DN'] = samsung['MA20'] - samsung['stddev'] * 2

kiwoom = kiwoom.sort_values(['일자'], ascending=True)
kiwoom['MA20'] = kiwoom['종가'].rolling(window=20).mean()
kiwoom['MA60'] = kiwoom['종가'].rolling(window=60).mean()
kiwoom['stddev'] = kiwoom['종가'].rolling(window=20).std()
kiwoom['Boll_UP'] = kiwoom['MA20'] + kiwoom['stddev'] * 2
kiwoom['Boll_DN'] = kiwoom['MA20'] - kiwoom['stddev'] * 2

samsung = samsung[60:].values
kiwoom = kiwoom[60:].values


def split_xy(dataset, time_step, y_col, target_col):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end = i + time_step
        y_end = x_end + y_col

        if y_end > len(dataset):
            break

        temp_x = dataset[i:x_end, :]
        temp_y = dataset[x_end:y_end, target_col]

        x.append(temp_x)
        y.append(temp_y)
    return np.array(x), np.array(y)


time_step = 5
samsung_cx, samsung_cy = split_xy(samsung, time_step, 2, 4)
kiwoom_cx, kiwoom_cy = split_xy(kiwoom, time_step, 2, 4)

samsung_cy = np.log1p(samsung_cy)
kiwoom_cy = np.log1p(kiwoom_cy)

sc_train_x, sc_test_x, sc_train_y, sc_test_y, kc_train_x, kc_test_x, kc_train_y, kc_test_y = train_test_split(
    samsung_cx, samsung_cy, kiwoom_cx, kiwoom_cy, train_size=0.8, shuffle=True, random_state=66)

sc_train_shape1 = sc_train_x.shape[1]
sc_train_shape2 = sc_train_x.shape[2]
sc_test_shape1 = sc_test_x.shape[1]
sc_test_shape2 = sc_test_x.shape[2]

sc_train_x = sc_train_x.reshape(sc_train_x.shape[0], sc_train_x.shape[1] * sc_train_x.shape[2])
sc_test_x = sc_test_x.reshape(sc_test_x.shape[0], sc_test_x.shape[1] * sc_test_x.shape[2])
kc_train_x = kc_train_x.reshape(kc_train_x.shape[0], kc_train_x.shape[1] * kc_train_x.shape[2])
kc_test_x = kc_test_x.reshape(kc_test_x.shape[0], kc_test_x.shape[1] * kc_test_x.shape[2])

scaler = MaxAbsScaler()
sc_train_x = scaler.fit_transform(sc_train_x)
sc_test_x = scaler.transform(sc_test_x)
kc_train_x = scaler.fit_transform(kc_train_x)
kc_test_x = scaler.transform(kc_test_x)

sc_train_x = sc_train_x.reshape(sc_train_x.shape[0], sc_train_shape1, sc_train_shape2)
sc_test_x = sc_test_x.reshape(sc_test_x.shape[0], sc_test_shape1, sc_test_shape2)

kc_train_x = kc_train_x.reshape(kc_train_x.shape[0], sc_train_shape1, sc_train_shape2)
kc_test_x = kc_test_x.reshape(kc_test_x.shape[0], sc_test_shape1, sc_test_shape2)

# 모델구성
input1 = Input(shape=(sc_train_x.shape[1], sc_train_x.shape[2]))
dense1_1 = LSTM(150, activation='relu', )(input1)
dense1_2 = Dense(70, activation='relu', )(dense1_1)
dense1_3 = Dense(70, activation='relu', )(dense1_2)
output1 = Dense(50, activation='relu', )(dense1_3)

input2 = Input(shape=(sc_train_x.shape[1], sc_train_x.shape[2]))
dense2_1 = LSTM(150, activation='relu', )(input2)
dense2_2 = Dense(70, activation='relu', )(dense2_1)
dense2_3 = Dense(70, activation='relu', )(dense2_2)
output2 = Dense(50, activation='relu', )(dense2_3)

from tensorflow.keras.layers import Concatenate

merge1 = Concatenate()([output1, output2])

output11 = Dense(70, activation='relu')(merge1)
output12 = Dense(50, activation='relu')(output11)
output13 = Dense(20, activation='relu')(output12)
output14 = Dense(10, activation='relu')(output13)
last_output1 = Dense(2, name='output--1')(output14)

output21 = Dense(70, activation='relu')(merge1)
output22 = Dense(50, activation='relu')(output21)
output23 = Dense(20, activation='relu')(output22)
output24 = Dense(10, activation='relu')(output23)
last_output2 = Dense(2, name='output--2')(output24)

model = Model(inputs=[input1, input2], outputs=[last_output1, last_output2])

# 3 컴파일0
epochs = 100000
patience = 500
model.compile(loss='mae', optimizer='adam', metrics=['mae'])
es = EarlyStopping(monitor='val_loss', patience=patience, mode='auto', restore_best_weights=True)
model.fit([sc_train_x, kc_train_x], [sc_train_y, kc_train_y], epochs=epochs, validation_split=0.2, callbacks=[es, mcp],
          batch_size=1)

loss = model.evaluate([sc_test_x, kc_test_x], [sc_test_y, kc_test_y], batch_size=1)

###### predict ##################
samsung_cx = samsung_cx.reshape(samsung_cx.shape[0], samsung_cx.shape[1] * samsung_cx.shape[2])
kiwoom_cx = kiwoom_cx.reshape(kiwoom_cx.shape[0], kiwoom_cx.shape[1] * kiwoom_cx.shape[2])

scaler = MaxAbsScaler()
samsung_cx = scaler.fit_transform(samsung_cx)
kiwoom_cx = scaler.fit_transform(kiwoom_cx)

samsung_cx = samsung_cx.reshape(samsung_cx.shape[0], sc_train_shape1, sc_train_shape2)
kiwoom_cx = kiwoom_cx.reshape(kiwoom_cx.shape[0], sc_train_shape1, sc_train_shape2)

sc_predict, kc_predict = model.predict([samsung_cx, kiwoom_cx])

samsung_1 = np.exp(sc_predict[-1][0])
kiwoom_1 = np.exp(kc_predict[-1][0])
samsung_2 = np.exp(sc_predict[-1][1])
kiwoom_2 = np.exp(kc_predict[-1][1])

samsung_str1 = format(int(samsung_1), ",d")
kiwoom_str1 = format(int(kiwoom_1), ",d")
samsung_str2 = format(int(samsung_2), ",d")
kiwoom_str2 = format(int(kiwoom_2), ",d")

print("=====================================")
print('Loss : ', loss)
print("=====================================")
print('월 삼성전자 거래량 예측 : ', samsung_1, f'({samsung_str1} 주)  vs 실제 11,170,845 주')
print('월 키움증권 거래량 예측 : ', kiwoom_1, f'({kiwoom_str1} 주)  vs 실제 49,658 주')
print('화 삼성전자 거래량 예측 : ', samsung_2, f'({samsung_str2} 주)')
print('화 키움증권 거래량 예측 : ', kiwoom_2, f'({kiwoom_str2} 주)')

'''
=====================================
Loss :  [0.5977494120597839, 0.24168327450752258, 0.35606613755226135, 0.24168327450752258, 0.35606613755226135]  
=====================================
월 삼성전자 거래량 예측 :  11803481.0 (11,803,481 주)  vs 실제 11,170,845 주
월 키움증권 거래량 예측 :  50637.516 (50,637 주)  vs 실제 49,658 주
화 삼성전자 거래량 예측 :  13417000.0 (13,417,000 주)
화 키움증권 거래량 예측 :  56068.05 (56,068 주)
'''



