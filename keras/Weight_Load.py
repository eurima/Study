import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MaxAbsScaler

# 1 데이터 & 모델
samsung = pd.read_csv("삼성전자.csv", encoding='cp949', index_col=0, header=0, sep=',', thousands=',')[:880]
kiwoom = pd.read_csv("키움증권.csv", encoding='cp949', index_col=0, header=0, sep=',', thousands=',')[:880]
model = load_model('sodam_volume.hdf5')

samsung = samsung.drop(['전일비', '등락률', 'Unnamed: 6', '금액(백만)', '신용비', '외국계', '프로그램', '외인비'], axis=1)
kiwoom = kiwoom.drop(['전일비', '등락률', 'Unnamed: 6', '금액(백만)', '신용비', '외국계', '프로그램', '외인비'], axis=1)

samsung = samsung.sort_values(['일자'], ascending=True)
samsung['MA20'] = samsung['종가'].rolling(window=20).mean()
samsung['MA60'] = samsung['종가'].rolling(window=60).mean()
samsung['stddef'] = samsung['종가'].rolling(window=20).std()
samsung['Boll_UP'] = samsung['MA20'] + samsung['stddef'] * 2
samsung['Boll_DN'] = samsung['MA20'] - samsung['stddef'] * 2

kiwoom = kiwoom.sort_values(['일자'], ascending=True)
kiwoom['MA20'] = kiwoom['종가'].rolling(window=20).mean()
kiwoom['MA60'] = kiwoom['종가'].rolling(window=60).mean()
kiwoom['stddef'] = kiwoom['종가'].rolling(window=20).std()
kiwoom['Boll_UP'] = kiwoom['MA20'] + kiwoom['stddef'] * 2
kiwoom['Boll_DN'] = kiwoom['MA20'] - kiwoom['stddef'] * 2

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

sc_train_shape1 = samsung_cx.shape[1]
sc_train_shape2 = samsung_cx.shape[2]

samsung_cx = samsung_cx.reshape(samsung_cx.shape[0], samsung_cx.shape[1] * samsung_cx.shape[2])
kiwoom_cx = kiwoom_cx.reshape(kiwoom_cx.shape[0], kiwoom_cx.shape[1] * kiwoom_cx.shape[2])

scaler = MaxAbsScaler()
samsung_cx = scaler.fit_transform(samsung_cx)
kiwoom_cx = scaler.fit_transform(kiwoom_cx)

samsung_cx = samsung_cx.reshape(samsung_cx.shape[0], sc_train_shape1, sc_train_shape2)
kiwoom_cx = kiwoom_cx.reshape(kiwoom_cx.shape[0], sc_train_shape1, sc_train_shape2)

#######################
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
print('월 삼성전자 거래량 예측 : ', samsung_1, f'({samsung_str1} 주)  vs 실제 11,170,845 주')
print('월 키움증권 거래량 예측 : ', kiwoom_1, f'({kiwoom_str1} 주)  vs 실제 49,658 주')
print('화 삼성전자 거래량 예측 : ', samsung_2, f'({samsung_str2} 주)')
print('화 키움증권 거래량 예측 : ', kiwoom_2, f'({kiwoom_str2} 주)')
'''
월 삼성전자 거래량 예측 :  11803481.0 (11,803,481 주)  vs 실제 11,170,845 주
월 키움증권 거래량 예측 :  50637.516 (50,637 주)  vs 실제 49,658 주
화 삼성전자 거래량 예측 :  13417000.0 (13,417,000 주)
화 키움증권 거래량 예측 :  56068.05 (56,068 주)
'''
