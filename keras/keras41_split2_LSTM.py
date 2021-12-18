import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

a = np.array(range(1,101))
size = 5 #x 4개 y1개

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1 ):
        subset = dataset[i:(i+size)]
        aaa.append(subset)
    return np.array(aaa)

dataset = split_x(a,size)
x = dataset[:,:-1]
y = dataset[:,-1]
#-----------------
# print(x,y,x.shape,y.shape)# (96, 4) (96,)
y_predict = np.array(range(96,106)) 
y_predict = split_x(y_predict,size)
# print(y_predict)
y_predict_x = y_predict[:,:-1]
# print(y_predict_x)
# # y_predict_y = y_predict[:,-1]
# print(y_predict_x.shape)#(6, 4)
x = x.reshape(96,4,1)
y_predict_x = y_predict_x.reshape(6,4,1)
#2 모델구성
model = Sequential()
model.add(LSTM(150,activation = 'relu', input_shape = (4,1)))
model.add(Dense(80, activation = 'relu'))
model.add(Dense(70, activation = 'relu'))
model.add(Dense(60, activation = 'relu'))
model.add(Dense(80, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(80, activation = 'relu'))
model.add(Dense(60, activation = 'relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(20, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1))
#3 컴파일
model.compile(loss='mse', optimizer = 'adam',metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=50, mode = 'auto', restore_best_weights=True)
model.fit(x, y, epochs = 1000, validation_split=0.2,callbacks=[es], batch_size = 1)
result = model.predict(y_predict_x)
print(result) 
'''      
[[ 99.98288 ]
 [100.980415]
 [101.97787 ]
 [102.975235]
 [103.97254 ]
 [104.9698  ]]
 -------------------------
 [[100.01418 ]
 [101.01451 ]
 [102.014824]
 [103.01506 ]
 [104.01534 ]
 [105.015625]]
'''
        