'''
각각의 Scaler의 특성과 정의 정리 해 놓을 것!!
1	StandardScaler	기본 스케일. 평균과 표준편차 사용

평균을 제거하고 데이터를 단위 분산으로 조정한다. 
그러나 이상치가 있다면 평균과 표준편차에 영향을 미쳐 변환된 데이터의 확산은 매우 달라지게 된다.
따라서 이상치가 있는 경우 균형 잡힌 척도를 보장할 수 없다.

2	MinMaxScaler	최대/최소값이 각각 1, 0이 되도록 스케일링

모든 feature 값이 0~1사이에 있도록 데이터를 재조정한다. 
다만 이상치가 있는 경우 변환된 값이 매우 좁은 범위로 압축될 수 있다.
즉, MinMaxScaler 역시 아웃라이어의 존재에 매우 민감하다.

3	MaxAbsScaler	최대절대값과 0이 각각 1, 0이 되도록 스케일링

절대값이 0~1사이에 매핑되도록 한다. 
즉 -1~1 사이로 재조정한다. 
양수 데이터로만 구성된 특징 데이터셋에서는 MinMaxScaler와 유사하게 동작하며, 큰 이상치에 민감할 수 있다.

4	RobustScaler	중앙값(median)과 IQR(interquartile range) 사용. 아웃라이어의 영향을 최소화

아웃라이어의 영향을 최소화한 기법이다. 중앙값(median)과 IQR(interquartile range)을 사용하기 때문에 
StandardScaler와 비교해보면 표준화 후 동일한 값을 더 넓게 분포 시키고 있음을 확인 할 수 있다.

'''

from os import scandir
from tensorflow.keras.models import Sequential , Model
from tensorflow.keras.layers import Dense, Input
import numpy as np
import time

# print(x.shape) 506,13
# print(y.shape) 506,

# print(dataset.feature_names)
# print(dataset.DESCR)

from sklearn.metrics import r2_score

#데이터
from sklearn.datasets import load_boston
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

dataset = load_boston()

x = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.8, shuffle = True, random_state = 66) 

# print(np.min(x),np.max(x)) #0 , 711.0
# x = x/np.max(x)  #<======== 전체가 적용된다 나쁘지는 않지만...

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = RobustScaler()
scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#[50, 20, 10, 50, 30, 15, 10, 5, 2]
deep_len = [100,80,60,40,50,80,70,60,50,40,30,20,10,5,4,2]
# model = Sequential() 
# model.add(Dense(deep_len[0], input_dim =x.shape[1])) 
# model.add(Dense(deep_len[1])) 
# model.add(Dense(deep_len[2]))
# model.add(Dense(deep_len[3])) 
# model.add(Dense(deep_len[4])) 
# model.add(Dense(deep_len[5])) 
# model.add(Dense(deep_len[6])) 
# model.add(Dense(deep_len[7])) 
# model.add(Dense(deep_len[8])) 
# model.add(Dense(deep_len[9])) 
# model.add(Dense(deep_len[10])) 
# model.add(Dense(deep_len[11])) 
# model.add(Dense(deep_len[12])) 
# model.add(Dense(deep_len[13])) 
# model.add(Dense(deep_len[14])) 
# model.add(Dense(deep_len[15])) 
# model.add(Dense(1))

input1 = Input(shape=(x.shape[1],))

dense1 = Dense(deep_len[0])(input1)
dense2 = Dense(deep_len[1])(dense1)
dense3 = Dense(deep_len[2])(dense2)
dense4 = Dense(deep_len[3])(dense3)
dense5 = Dense(deep_len[4])(dense4)
dense6 = Dense(deep_len[5])(dense5)
dense7 = Dense(deep_len[6])(dense6)
dense8 = Dense(deep_len[7])(dense7)
dense9 = Dense(deep_len[8])(dense8)
dense10 = Dense(deep_len[9])(dense9)
dense11 = Dense(deep_len[10])(dense10)
dense12 = Dense(deep_len[11])(dense11)
dense13 = Dense(deep_len[12])(dense12)
dense14 = Dense(deep_len[13])(dense13)
dense15 = Dense(deep_len[14])(dense14)
dense16 = Dense(deep_len[15])(dense15)
output1 = Dense(1)(dense16)

model = Model(inputs = input1, outputs = output1)


#3. 컴파일, 훈련
epoch = 1000
model.compile(loss = 'mse', optimizer = 'adam')
from tensorflow.keras.callbacks import EarlyStopping
patience_num = 50
es = EarlyStopping(monitor='val_loss', patience=patience_num, mode = 'auto', verbose=1, restore_best_weights=True)
start = time.time()

model.fit(x_train, y_train, epochs = epoch, validation_split=0.2,callbacks=[es], batch_size =1)#batch_size =32 가 default
end = time.time() - start


#4 평가예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print("R2 : ",r2)
print(deep_len)
print("epochs :",epoch)

acc= str(loss[1]).replace(".", "_")
model.save(f"./_save/boston_{acc}.h5")
'''
Normal
Epoch 00263: early stopping
4/4 [==============================] - 0s 988us/step - loss: 18.3059
loss :  18.305864334106445
R2 :  0.7809856267013906

MinMaxScaler
Epoch 00091: early stopping
4/4 [==============================] - 0s 1ms/step - loss: 17.2960
loss :  17.295957565307617
R2 :  0.7930683248039497


StandardScaler
Epoch 00097: early stopping
4/4 [==============================] - 0s 1ms/step - loss: 16.7891
loss :  16.78911781311035
R2 :  0.7991322361793805

RobustScaler
Epoch 00075: early stopping
4/4 [==============================] - 0s 989us/step - loss: 20.6809
loss :  20.680896759033203
R2 :  0.752570369378124


MaxAbsScaler
Epoch 00074: early stopping
4/4 [==============================] - 0s 665us/step - loss: 16.4644
loss :  16.46441078186035
R2 :  0.8030170670197926

Epoch 00103: early stopping
4/4 [==============================] - 0s 737us/step - loss: 18.6164
loss :  18.616403579711914
R2 :  0.7772702820559952


'''
