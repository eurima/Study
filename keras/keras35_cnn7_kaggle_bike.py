from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import time
import pandas as pd

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
# x = train.drop(['casual','registered'], axis =1) #--->상관관계분석용
x = train.drop(['casual','registered','count'], axis =1) #

x['datetime'] = pd.to_datetime(x['datetime'])
x['year'] = x['datetime'].dt.year
x['month'] = x['datetime'].dt.month
x['day'] = x['datetime'].dt.day
x['hour'] = x['datetime'].dt.hour
x = x.drop('datetime', axis=1)
'''
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,10))
sns.heatmap(data= x.corr(), square=True, annot=True, cbar=True)
plt.show()
'''
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

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n = x_train.shape[0]
x_train_transe = scaler.fit_transform(x_train) 
print(x_train_transe.shape) #8708,12

x_train = x_train_transe.reshape(n,2,2,3) 
m = x_test.shape[0]
x_test = scaler.transform(x_test.reshape(m,-1)).reshape(m,2,2,3)

model = Sequential()
model.add(Conv2D(128, kernel_size=(4,4),padding ='same',strides=1, 
                 input_shape = (x_train.shape[1],x_train.shape[2],x_train.shape[3])))#
model.add(MaxPooling2D())
model.add(Conv2D(64,(2,2),padding ='same', activation='relu'))#<------------
# model.add(MaxPooling2D())
# model.add(Dropout(0.2))
model.add(Conv2D(32,(2,2),padding ='same', activation='relu'))
# model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(10))
# model.add(Dropout(0.2))
model.add(Dense(5))
model.add(Dense(2))
model.add(Dropout(0.5))
model.add(Dense(1))

#3. 컴파일, 훈련
########################################################################
model.compile(loss = 'mse', optimizer = 'adam')

#3. 컴파일, 훈련
# model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy']) 
########################################################################
start = time.time()
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
epoch = 10000
patience_num = 500
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")
filepath = "./_ModelCheckPoint/"
filename = '{epoch:04d}-{val_loss:4f}.hdf5' #filepath + datetime
model_path = "".join([filepath,'k35_cnn6_fetch_',datetime,"_",filename])
es = EarlyStopping(monitor='val_loss', patience=patience_num, mode = 'auto', restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose=1, save_best_only= True, filepath = model_path)
hist = model.fit(x_train, y_train, epochs = epoch, validation_split=0.2, callbacks=[es,mcp], batch_size = 50)
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

'''
<DNN>
print("loss : ",loss[0]) 
print("accuracy : ",loss[1])
<Relu>
Epoch 00281: early stopping
시간 :  77.48 초
69/69 [==============================] - 0s 543us/step - loss: 0.1747
loss :  0.1746547520160675
R2 :  0.9108701588449649
RMSE :  0.417917188949065


<<CNN>>
시간 :  171.88 초
69/69 [==============================] - 0s 587us/step - loss: 0.2252
loss :  0.22520200908184052
R2 :  0.8850748909933529
RMSE :  0.47455453713341206

'''
