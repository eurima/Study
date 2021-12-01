from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston, load_diabetes
import time

#1.데이터
dataset = load_diabetes()
x = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.8, shuffle = True, random_state = 66) 

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = RobustScaler()
scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

deep_len = [200,150,100,80,70,60,50,40,30,20,10,5,4,3,2,2]

model = Sequential() 
model.add(Dense(deep_len[0], input_dim = 10)) 
model.add(Dense(deep_len[1])) 
model.add(Dense(deep_len[2]))
model.add(Dense(deep_len[3])) 
model.add(Dense(deep_len[4])) 
model.add(Dense(deep_len[5])) 
model.add(Dense(deep_len[6])) 
model.add(Dense(deep_len[7])) 
# model.add(Dense(deep_len[8])) 
# model.add(Dense(deep_len[9])) 
# model.add(Dense(deep_len[10])) 
# model.add(Dense(deep_len[11])) 
# model.add(Dense(deep_len[12])) 
# model.add(Dense(deep_len[13])) 
# model.add(Dense(deep_len[14])) 
# model.add(Dense(deep_len[15])) 
model.add(Dense(1)) 

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
patience_num = 50
es = EarlyStopping(monitor='val_loss', patience=patience_num, mode = 'min', verbose=1)
epoch = 10000
model.compile(loss = 'mse', optimizer = 'adam')
start = time.time()
hist = model.fit(x_train, y_train, epochs = epoch, batch_size =1,validation_split=0.2,callbacks=[es])


end = time.time() - start
print('시간 : ', round(end,2) ,'초')

#4 평가예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print("R2 : ",r2)
print(deep_len)
print("epochs :",epoch)


'''
Normal 

Epoch 00103: early stopping
시간 :  24.73 초
3/3 [==============================] - 0s 0s/step - loss: 3334.9238
loss :  3334.923828125
r2 :  0.4861474526091143

MinMaxScaler
Epoch 00059: early stopping
시간 :  12.34 초
3/3 [==============================] - 0s 999us/step - loss: 3194.6562
loss :  3194.65625
R2 :  0.5077602072769352

StandardScaler
Epoch 00095: early stopping
시간 :  19.5 초
3/3 [==============================] - 0s 993us/step - loss: 3326.4336
loss :  3326.43359375
R2 :  0.4874556638004163

RobustScaler
Epoch 00056: early stopping
시간 :  11.45 초
3/3 [==============================] - 0s 1ms/step - loss: 3311.6853
loss :  3311.685302734375
R2 :  0.4897280693156646

MaxAbsScaler
Epoch 00054: early stopping
시간 :  11.68 초
3/3 [==============================] - 0s 983us/step - loss: 3256.4846
loss :  3256.484619140625
R2 :  0.4982335197352715






'''

