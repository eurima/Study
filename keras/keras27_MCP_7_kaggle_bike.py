import time
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

import matplotlib.pyplot as plt

def RMSE(y_test, y_predict):
      return np.sqrt(mean_squared_error(y_test, y_predict))  

path = "D:\\Study\\_data\\bike\\"
train = pd.read_csv(path +"train.csv")
# print(train) 10886,12

test_flie = pd.read_csv(path + "test.csv") #### 제출용 test는 시험용!
# print(test) 6493,9
submission = pd.read_csv(path+"sampleSubmission.csv") #제출할 값
# print(submission) 6493,2

y = train['count']
x = train.drop(['casual','registered','count'], axis =1) #

x['datetime'] = pd.to_datetime(x['datetime'])
x['year'] = x['datetime'].dt.year
x['month'] = x['datetime'].dt.month
x['day'] = x['datetime'].dt.day
x['hour'] = x['datetime'].dt.hour
x = x.drop('datetime', axis=1)


test_flie['datetime'] = pd.to_datetime(test_flie['datetime'])
test_flie['year'] = test_flie['datetime'].dt.year
test_flie['month'] = test_flie['datetime'].dt.month
test_flie['day'] = test_flie['datetime'].dt.day
test_flie['hour'] = test_flie['datetime'].dt.hour
test_flie = test_flie.drop('datetime', axis=1)


# 로그변환
y = np.log1p(y) #----------> 로그변환 하면서 +1 해줌

x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.8, shuffle = True, random_state = 66) #

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = MaxAbsScaler()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)

#2 모델구성#        
deep_len = [100, 80, 60, 50, 40, 80, 70, 60, 50, 40, 30, 20, 10, 5, 4, 2]
model = Sequential()
model.add(Dense(deep_len[0], input_dim =x.shape[1]))
model.add(Dense(deep_len[1], activation = 'linear' )) 
model.add(Dense(deep_len[2]))
model.add(Dense(deep_len[3])) 
model.add(Dense(deep_len[4])) 
model.add(Dense(deep_len[5], activation = 'relu'))
model.add(Dense(deep_len[6])) 
model.add(Dense(deep_len[7] )) 
model.add(Dense(deep_len[8])) 
model.add(Dense(deep_len[9])) 
model.add(Dense(deep_len[10], activation = 'relu'))
model.add(Dense(deep_len[11])) 
model.add(Dense(deep_len[12])) 
model.add(Dense(deep_len[13])) 
model.add(Dense(deep_len[14])) 
model.add(Dense(deep_len[15])) 
model.add(Dense(1))

#3. 컴파일, 훈련
########################################################################
model.compile(loss = 'mse', optimizer = 'adam')
start = time.time()
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
epoch = 10000
patience_num = 100
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")
filepath = "./_ModelCheckPoint/"
filename = '{epoch:04d}-{val_loss:4f}.hdf5' #filepath + datetime
model_path = "".join([filepath,'k27_bike_',datetime,"_",filename])
es = EarlyStopping(monitor='val_loss', patience=patience_num, mode = 'auto', restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose=1, save_best_only= True, filepath = model_path)
model.fit(x_train, y_train, epochs = epoch, validation_split=0.2,callbacks=[es,mcp], batch_size =1)
end = time.time() - start
print('시간 : ', round(end,2) ,'초')
########################################################################
#4 평가예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss) 

y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)

rmse = RMSE(y_test, y_predict)
print("R2 : ", r2)
print("RMSE : ",rmse)
print("epochs :",epoch)
print(deep_len)

'''
<<기존 성과 우수>>
========== StandardScaler
Epoch 00104: early stopping
시간 :  26.44 초
69/69 [==============================] - 0s 493us/step - loss: 1.0677
loss :  1.0677317380905151
R2 :  0.45511496145529395
RMSE :  1.0333111350166428
<Relu>
Epoch 00281: early stopping
시간 :  77.48 초
69/69 [==============================] - 0s 543us/step - loss: 0.1747
loss :  0.1746547520160675
R2 :  0.9108701588449649
RMSE :  0.417917188949065

<<ModelCheckpoint >>
Epoch 00137: val_loss did not improve from 1.94779
시간 :  755.55 초
69/69 [==============================] - 0s 772us/step - loss: 1.9603
loss :  1.9602636098861694
R2 :  -0.0003617055476405895
RMSE :  1.400094007994115



'''
scaler.transform(test_flie)
############### 제출용.
result = model.predict(test_flie)
submission['count'] = result

# print(submission[:10])
submission.to_csv(path+"Relu_sampleHR_StandardScaler_MCK.csv", index = False)
