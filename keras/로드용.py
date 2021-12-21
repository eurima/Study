import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

#1 데이터 & 모델
samsung = pd.read_csv("삼성전자.csv", encoding='cp949', index_col = 0, header = 0, sep = ',', thousands=',')[:880]
kiwoom = pd.read_csv("키움증권.csv", encoding='cp949', index_col = 0,header = 0, sep = ',', thousands=',')[:880]
model = load_model('소담수욜.hdf5')

samsung = samsung.drop(['전일비', '등락률', 'Unnamed: 6', '금액(백만)', '신용비', '외국계', '프로그램', '외인비'], axis=1)
kiwoom = kiwoom.drop(['전일비', '등락률', 'Unnamed: 6', '금액(백만)', '신용비', '외국계', '프로그램', '외인비'], axis=1)

samsung = samsung.sort_values(['일자'], ascending=True)
kiwoom = kiwoom.sort_values(['일자'], ascending=True)

samsung['MA20'] = samsung['종가'].rolling(window=20).mean()
samsung['MA60'] = samsung['종가'].rolling(window=60).mean()
samsung['stddev'] = samsung['종가'].rolling(window=20).std()
samsung['Boll_UP'] = samsung['MA20'] + samsung['stddev']*2
samsung['Boll_DN'] = samsung['MA20'] - samsung['stddev']*2

kiwoom['MA20'] = kiwoom['종가'].rolling(window=20).mean()
kiwoom['MA60'] = kiwoom['종가'].rolling(window=60).mean()
kiwoom['stddev'] = kiwoom['종가'].rolling(window=20).std()
kiwoom['Boll_UP'] = kiwoom['MA20'] + kiwoom['stddev']*2
kiwoom['Boll_DN'] = kiwoom['MA20'] - kiwoom['stddev']*2

samsung = samsung[-60:].values
kiwoom = kiwoom[-60:].values

def split_xy(dataset, time_steps, y_column, target_col):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps  # 0 + 5 = 5
        y_end_number = x_end_number + y_column  # 5 + 2 -0 = 7
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number,:-1]  # 0 :
        tmp_y = dataset[x_end_number :y_end_number, target_col]  # 5 : 7 , 0
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

time_step = 5
y_col = 3 #3 ->53, 4 53,5,12 / 2-> 54,3 54,5,12
target_col = 0
samsung_cx, samsung_cy = split_xy(samsung, time_step, y_col, target_col)
kiwoom_cx, kiwoom_cy = split_xy(kiwoom, time_step, y_col, target_col)

sc_train_shape1 = samsung_cx.shape[1]
sc_train_shape2 = samsung_cx.shape[2]

samsung_cx  = samsung_cx.reshape(samsung_cx.shape[0],samsung_cx.shape[1]*samsung_cx.shape[2])
kiwoom_cx  = kiwoom_cx.reshape(kiwoom_cx.shape[0],kiwoom_cx.shape[1]*kiwoom_cx.shape[2])

scaler = MaxAbsScaler()
samsung_cx = scaler.fit_transform(samsung_cx)
kiwoom_cx = scaler.fit_transform(kiwoom_cx)

samsung_cx  = samsung_cx.reshape(samsung_cx.shape[0],sc_train_shape1,sc_train_shape2)
kiwoom_cx  = kiwoom_cx.reshape(kiwoom_cx.shape[0],sc_train_shape1,sc_train_shape2)

#######################
sc_predict, kc_predict = np.exp(model.predict([samsung_cx, kiwoom_cx]))

print(sc_predict[-1])
print(kc_predict[-1])

samsung_open_wed = sc_predict[-1][2]
kiwoom_open_wed = kc_predict[-1][2]

samsung_open_wed_str = format(round(int(samsung_open_wed),-2), ",d")
kiwoom_open_wed_str = format(round(int(kiwoom_open_wed),-2), ",d")

print('수 삼성전자 시가 예측 : ', samsung_open_wed, f'({samsung_open_wed_str} 원)')
print('수 키움증권 시가 예측 : ', kiwoom_open_wed, f'({kiwoom_open_wed_str} 원)')
