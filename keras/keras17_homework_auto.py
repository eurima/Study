from sklearn.datasets import fetch_covtype
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
import numpy as np
import time

#1 데이터
dataset  = fetch_covtype()
# print(dataset)
# print(dataset.DESCR) 
'''
    =================   ============
    Classes                        7
    Samples total             581012
    Dimensionality                54
    Features                     int
    =================   ============
'''
''''''

x = dataset.data
y = dataset.target #===== sklearn에서만 제공!!
# print(x.shape, y.shape)  #(581012, 54) (581012,)
# print(np.unique(y)) #---->  배열의 고유값을 찾아준다 (라벨값이 어떤것이 있는가) len(np.unique(y))

activation_fun =""
loss = ""

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
# print(y.shape)

if len(np.unique(y)) > 2:
    activation_fun = 'softmax'
    loss = 'categorical_crossentropy'
    #========== to_categorical ===============
    y = to_categorical(y) 
    
    # print(y)#[5 5 2 ... 3 3 3]    
    #========== OneHotEncoder ===============
    #one_y = OneHotEncoder()    
    #one_y.fit(y.reshape(-1,1))
    #y  = one_y.transform(y.reshape(-1,1)).toarray()
    # print(y[0]) [0. 0. 0. 0. 1. 0. 0.]
    # print(y.shape) #581012,7    
    
    #========== pandas get_dummies ===============
    # import pandas as pd
    # y = pd.get_dummies(y)
    #print(y.shape) #(581012, 7)    
    
    print("====================== Categorical =================")
else:
    activation_fun = 'sigmoid'
    loss = 'binary_crossentropy'
    print("====================== Binary =================")

# time.sleep(5)  

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.8, shuffle = True, random_state = 66) #
#2 모델구성#        
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
model.add(Dense(y.shape[1], activation = activation_fun))


#3. 컴파일, 훈련
epoch = 10000
model.compile(loss = loss, optimizer = 'adam', metrics=['accuracy']) 

from tensorflow.keras.callbacks import EarlyStopping
patience_num = 50
es = EarlyStopping(monitor='val_loss', patience=patience_num, mode = 'auto', verbose=1, restore_best_weights=True)
start = time.time()

model.fit(x_train, y_train, epochs = epoch, validation_split=0.2,callbacks=[es],batch_size = 100)#batch_size =10, 
end = time.time() - start
print('시간 : ', round(end,2) ,'초')

#4 평가예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss[0]) 
print("accuracy : ",loss[1])

y_predict = model.predict(x_test)
print("epochs :",epoch)
''' batch_size = 100
==== OneHotEncoder 사용
print(y[0]) [0. 0. 0. 0. 1. 0. 0.]
print(y.shape) #(581012,7)

3719/3719 [==============================] - 5s 1ms/step - loss: 0.6718 - accuracy: 0.7144 - val_loss: 0.6721 - val_accuracy: 0.7117
Restoring model weights from the end of the best epoch.
Epoch 00233: early stopping
시간 :  1134.0 초
3632/3632 [==============================] - 2s 498us/step - loss: 0.6636 - accuracy: 0.7184
loss :  0.6635791659355164
accuracy :  0.7184152007102966

==== Pandas get_dummies 사용
print(y.shape) #(581012, 7)
Epoch 304/10000
3719/3719 [==============================] - 5s 1ms/step - loss: 0.6715 - accuracy: 0.7161 - val_loss: 0.6650 - val_accuracy: 0.7193
Restoring model weights from the end of the best epoch.
Epoch 00304: early stopping
시간 :  1477.16 초
3632/3632 [==============================] - 2s 490us/step - loss: 0.6637 - accuracy: 0.7211
loss :  0.6636537909507751
accuracy :  0.721117377281189

==== to_categorical 사용
3719/3719 [==============================] - 6s 2ms/step - loss: 0.6735 - accuracy: 0.7118 - val_loss: 0.6729 - val_accuracy: 0.7128
Restoring model weights from the end of the best epoch.
Epoch 00172: early stopping
시간 :  867.49 초
3632/3632 [==============================] - 2s 581us/step - loss: 0.6654 - accuracy: 0.7148
loss :  0.6653769016265869
accuracy :  0.7147663831710815












'''


'''
3719/3719 [==============================] - 5s 1ms/step - loss: 0.6734 - accuracy: 0.7159 - val_loss: 0.6758 - val_accuracy: 0.7121
Restoring model weights from the end of the best epoch.
Epoch 00239: early stopping
시간 :  1183.49 초
3632/3632 [==============================] - 2s 502us/step - loss: 0.6658 - accuracy: 0.7203
loss :  0.665805995464325
accuracy :  0.7202998399734497
batch_size =100

Epoch 166/10000
37185/37185 [==============================] - 36s 958us/step - loss: 0.6827 - accuracy: 0.7116 - val_loss: 0.6844 - val_accuracy: 0.7128
Restoring model weights from the end of the best epoch.
Epoch 00166: early stopping
시간 :  5986.25 초
batch_size =10

Epoch 191/10000
11621/11621 [==============================] - 12s 1ms/step - loss: 0.6773 - accuracy: 0.7136 - val_loss: 0.6772 - val_accuracy: 0.7140    
Restoring model weights from the end of the best epoch.
Epoch 00191: early stopping
시간 :  2413.25 초
3632/3632 [==============================] - 2s 499us/step - loss: 0.6676 - accuracy: 0.7203
loss :  0.667597234249115
accuracy :  0.7202568054199219
batch_size = default


https://keras.io/api/models/model_training_apis/
batch_size: Integer or None. Number of samples per gradient update. If unspecified, batch_size will default to 32.
'''
