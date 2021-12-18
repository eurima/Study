from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
from sklearn.metrics import r2_score
import time

dataset  = load_breast_cancer()
# print(dataset)
# print(dataset.DESCR)
# print(dataset.feature_name)

x = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.8, shuffle = True, random_state = 66) 
from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)

deep_len = [100,80,60,40,50,80,70,60,50,40,30,20,10,5,4,2]

model = Sequential() 
model.add(Conv1D(150,2,activation = 'relu', input_shape = (x_train.shape[1],x_train.shape[2])))
model.add(Flatten())
model.add(Dense(deep_len[1])) 
model.add(Dense(deep_len[2]))
model.add(Dense(deep_len[3])) 
model.add(Dense(deep_len[4])) 
model.add(Dense(deep_len[5])) 
model.add(Dense(deep_len[6])) 
model.add(Dense(deep_len[7])) 
model.add(Dense(deep_len[8])) 
model.add(Dense(deep_len[9])) 
model.add(Dense(deep_len[10])) 
model.add(Dense(deep_len[11])) 
model.add(Dense(deep_len[12])) 
model.add(Dense(deep_len[13])) 
model.add(Dense(deep_len[14])) 
model.add(Dense(deep_len[15])) 
model.add(Dense(1, activation = 'sigmoid'))

########################################################################
model.compile(loss = 'mse', optimizer = 'adam', metrics=['mae'])
start = time.time()
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
epoch = 10000
patience_num = 500
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")
filepath = "./_ModelCheckPoint/"
filename = '{epoch:04d}-{val_loss:4f}.hdf5' #filepath + datetime
model_path = "".join([filepath,'k27_boston_',datetime,"_",filename])
es = EarlyStopping(monitor='val_loss', patience=patience_num, mode = 'auto', restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose=1, save_best_only= True, filepath = model_path)
model.fit(x_train, y_train, epochs = epoch, validation_split=0.2,callbacks=[es,mcp], batch_size =1000)
end = time.time() - start
########################################################################

#4 평가예측
loss = model.evaluate(x_test,y_test)
print('시간 : ', round(end,2) ,'초')
print("loss : ",loss[0]) 
print("accuracy : ",loss[1])

y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print("R2 : ",r2)
    
'''
Normal
Epoch 00164: early stopping
시간 :  51.66 초
4/4 [==============================] - 0s 997us/step - loss: 0.2165 - accuracy: 0.9298
loss :  0.21648594737052917
accuracy :  0.9298245906829834

MinMaxScaler
Epoch 00161: early stopping
시간 :  52.22 초
4/4 [==============================] - 0s 997us/step - loss: 0.2865 - accuracy: 0.9298
loss :  0.2864888310432434
accuracy :  0.9298245906829834


StandardScaler
Epoch 00308: early stopping
시간 :  102.06 초
4/4 [==============================] - 0s 998us/step - loss: 0.1533 - accuracy: 0.9298
loss :  0.15333078801631927
accuracy :  0.9298245906829834

RobustScaler
Epoch 00262: early stopping
시간 :  84.39 초
4/4 [==============================] - 0s 997us/step - loss: 0.1608 - accuracy: 0.9298
loss :  0.1608397215604782
accuracy :  0.9298245906829834


MaxAbsScaler
Epoch 00127: early stopping
시간 :  41.91 초
4/4 [==============================] - 0s 998us/step - loss: 0.2551 - accuracy: 0.9123
loss :  0.25514113903045654
accuracy :  0.9122806787490845

Epoch 00205: early stopping
시간 :  64.3 초
4/4 [==============================] - 0s 997us/step - loss: 0.2059 - accuracy: 0.9298
loss :  0.20594066381454468
accuracy :  0.9298245906829834


===================== LSTM ==============================================
loss :  0.02884048782289028
accuracy :  0.9736841917037964
R2 :  0.8747708062969191

=============== CONV1D ==================
시간 :  16.34 초
loss :  0.028067149221897125
accuracy :  0.9736841917037964
R2 :  0.878128750115488


'''
