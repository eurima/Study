from sklearn.datasets import fetch_covtype
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import time

#1 데이터
dataset  = fetch_covtype()
# print(dataset)
# print(dataset.DESCR) 

x = dataset.data
y = dataset.target #===== sklearn에서만 제공!!
# print(x.shape, y.shape) 
# print(np.unique(y)) #---->  배열의 고유값을 찾아준다 (라벨값이 어떤것이 있는가) len(np.unique(y))

from tensorflow.keras.utils import to_categorical
y = to_categorical(y) 

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.8, shuffle = True, random_state = 66) #
#2 모델구성
#        
deep_len = [100, 50, 30, 20, 100, 50, 30, 40, 50, 40, 30, 20, 10, 5, 4, 2]
model = Sequential()
model.add(Dense(deep_len[0], activation = 'linear', input_dim =x.shape[1]))
model.add(Dense(deep_len[1], )) 
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
model.add(Dense(y.shape[1], activation = 'softmax'))


#3. 컴파일, 훈련
epoch = 10000
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy']) 

from tensorflow.keras.callbacks import EarlyStopping
patience_num = 50
es = EarlyStopping(monitor='val_loss', patience=patience_num, mode = 'auto', verbose=1, restore_best_weights=True)
start = time.time()

model.fit(x_train, y_train, epochs = epoch, batch_size =100, validation_split=0.2,callbacks=[es])
end = time.time() - start
print('시간 : ', round(end,2) ,'초')

#4 평가예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss[0]) 
print("accuracy : ",loss[1])

y_predict = model.predict(x_test)
print("epochs :",epoch)
# result = y_predict[:7]
# print(result)
# print(y_test[:7])
'''

3719/3719 [==============================] - 5s 1ms/step - loss: 0.6734 - accuracy: 0.7159 - val_loss: 0.6758 - val_accuracy: 0.7121
Restoring model weights from the end of the best epoch.
Epoch 00239: early stopping
시간 :  1183.49 초
3632/3632 [==============================] - 2s 502us/step - loss: 0.6658 - accuracy: 0.7203
loss :  0.665805995464325
accuracy :  0.7202998399734497

batch_size =100

'''
