import time
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential , Model
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten,Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

def RMSE(y_test, y_predict):
      return np.sqrt(mean_squared_error(y_test, y_predict))  
path = "D:\\Study\\_data\\bike\\"
train = pd.read_csv(path +"train.csv")
test_flie = pd.read_csv(path + "test.csv") #### 제출용 test는 시험용!
submission = pd.read_csv(path+"sampleSubmission.csv") #제출할 값
# print(submission) 6493,2

y = train['count']
x = train.drop(['casual','registered','count'], axis =1) #
# test_flie = test_flie.drop(['casual','registered','count'], axis =1) #

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
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.8, shuffle = True, random_state = 66) 

from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_flie = scaler.transform(test_flie)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
test_flie = test_flie.reshape(test_flie.shape[0],test_flie.shape[1],1)

#2 모델구성#        
deep_len = [100, 80, 60, 50, 40, 80, 70, 60, 50, 40, 30, 20, 10, 5, 4, 2]
model = Sequential() 
# model.add(LSTM(150,activation = 'relu', input_shape = (x_train.shape[1],x_train.shape[2])))
model.add(Conv1D(150,2,activation = 'relu', input_shape = (x_train.shape[1],x_train.shape[2])))
model.add(Flatten())
# model.add(Dense(deep_len[1],activation = 'relu',input_dim=x_train.shape[1])) 
model.add(Dense(deep_len[2],activation = 'relu',))
model.add(Dropout(0.2))
model.add(Dense(deep_len[3],activation = 'relu',)) 
model.add(Dense(deep_len[4],activation = 'relu',)) 
model.add(Dense(deep_len[5],activation = 'relu',)) 
model.add(Dropout(0.2))
model.add(Dense(deep_len[6],activation = 'relu',)) 
model.add(Dense(deep_len[7],activation = 'relu',)) 
model.add(Dense(deep_len[8],activation = 'relu',)) 
model.add(Dense(deep_len[9],activation = 'relu',)) 
model.add(Dense(deep_len[10],activation = 'relu',)) 
model.add(Dense(deep_len[11],activation = 'relu',)) 
model.add(Dense(deep_len[12],activation = 'relu',)) 
model.add(Dense(deep_len[13],activation = 'relu',)) 
model.add(Dense(deep_len[14],activation = 'relu',)) 
model.add(Dense(deep_len[15],activation = 'relu',)) 
model.add(Dropout(0.2))
model.add(Dense(1))

#3. 컴파일, 훈련
epoch = 100000
model.compile(loss = "mse", optimizer = 'adam')
from tensorflow.keras.callbacks import EarlyStopping
patience_num = 50
es = EarlyStopping(monitor='val_loss', patience=patience_num, mode = 'auto', verbose=1, restore_best_weights=True)
start = time.time()

model.fit(x_train, y_train, epochs = epoch, validation_split=0.2,callbacks=[es], batch_size =1)#
end = time.time() - start

#4 평가예측
loss = model.evaluate(x_test,y_test)
print('시간 : ', round(end,2) ,'초')
print("loss : ",loss) 

y_predict = model.predict(x_test)
# print(y_predict)
r2 = r2_score(y_test,y_predict)
rmse = RMSE(y_test, y_predict)
print("R2 : ", r2)
print("RMSE : ",rmse)
# model.save(f"./_save/kaggle_bike_{r2}.h5")
############### 제출용.
result = model.predict(test_flie)
result = np.expm1(result).astype(int)
submission['count'] = result
print(submission[:10])
submission.to_csv(path+f"{r2}Bike.csv", index = False)
