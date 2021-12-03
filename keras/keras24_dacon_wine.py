from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, LabelEncoder

#1 데이터
# dataset  = load_wine()
# print(dataset)
# print(dataset.DESCR) 

# x = dataset.data
# y = dataset.target #===== sklearn에서만 제공!!
# print(x.shape, y.shape) 
# print(np.unique(y)) #---->  배열의 고유값을 찾아준다 (라벨값이 어떤것이 있는가) len(np.unique(y))
path = "D:\\_data\\dacon\\wine\\" #"D:\\Study\\_data\\wine\\" 
train = pd.read_csv(path +"train.csv")
# print(train) 10886,12

test_flie = pd.read_csv(path + "test.csv") #### 제출용 test는 시험용!
# print(test) 6493,9
submission = pd.read_csv(path+"sample_Submission.csv") #제출할 값
y = train['quality']
x = train.drop(['quality'], axis =1) #
# x = train #.drop(['casual','registered','count'], axis =1) #

le = LabelEncoder()
le.fit(train.type)
x_type = le.transform(train['type'])
# x = x.drop(['type'], axis = 1)
# x = pd.concat([x,x_type])
x['type'] = x_type
# print(x)

from tensorflow.keras.utils import to_categorical
# one_hot = to_categorical(y,num_classes=len(np.unique(y)))
y = to_categorical(y) #<=============== class 개수대로 자동으로 분류 해 준다!!! /// 간단!!

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.8, shuffle = True, random_state = 66) #

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = RobustScaler()
scaler = MaxAbsScaler()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)



#2 모델구성
#        
deep_len = [100, 50, 30, 20, 100, 50, 30, 40, 50, 40, 30, 20, 10, 5, 4, 2]
model = Sequential()
model.add(Dense(deep_len[0], activation = 'linear', input_dim =x.shape[1]))
model.add(Dense(deep_len[1], )) # ===> 디폴트 값은 linear이고 sigmoid를 넣을 수도 있다 (값이 튀다면 sigmoid로 한번씩 잡아주면 성능이 좋아질 수 있다)
model.add(Dense(deep_len[2]))
model.add(Dense(deep_len[3])) 
model.add(Dense(deep_len[4])) 
model.add(Dense(deep_len[5],activation ='relu'))
model.add(Dense(deep_len[6])) 
model.add(Dense(deep_len[7])) 
model.add(Dense(deep_len[8])) 
model.add(Dense(deep_len[9])) 
model.add(Dense(deep_len[10],activation ='relu'))
model.add(Dense(deep_len[11])) 
model.add(Dense(deep_len[12])) 
model.add(Dense(deep_len[13])) 
model.add(Dense(deep_len[14])) 
model.add(Dense(deep_len[15])) 
model.add(Dense(y.shape[1])) #이진분류의 마지막 레이어는 무조건 sigmoid!!!!
# sigmoid는 0 ~ 1 사이의 값을 뱉는다

#3. 컴파일, 훈련
epoch = 10000
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy']) # metrics=['accuracy'] 영향을 미치지 않는다

from tensorflow.keras.callbacks import EarlyStopping
patience_num = 50
es = EarlyStopping(monitor='val_loss', patience=patience_num, mode = 'auto', verbose=1, restore_best_weights=True)
# 통상 val_loss 가 성능이 더 좋다
# 그렇지만 너무 튄다면 loss를 넣어도 된다
# 나중에는 monitor='accuracy' ,mode 가 헷갈리면 'auto'로 잡는다
start = time.time()

model.fit(x_train, y_train, epochs = epoch, batch_size =1,validation_split=0.2,callbacks=[es])
end = time.time() - start
print('시간 : ', round(end,2) ,'초')

#4 평가예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss[0]) #<==== List 형태로 제공된다
print("accuracy : ",loss[1])
print("epochs :",epoch)



le.fit(test_flie.type)
test_flie_type = le.transform(test_flie['type'])
# x = x.drop(['type'], axis = 1)
# x = pd.concat([x,x_type])
test_flie['type'] = test_flie_type
# y_predict = model.predict(x_test)

print(test_flie)

scaler.transform(test_flie)
# ############### 제출용.
result = model.predict(test_flie)
submission['quality'] = result

# # print(submission[:10])
submission.to_csv(path+"sampleHR_MaxAbsScaler.csv", index = False)








'''
Normal
Epoch 00277: early stopping
시간 :  30.65 초
2/2 [==============================] - 0s 2ms/step - loss: 0.2468 - accuracy: 0.8889
loss :  0.24677377939224243
accuracy :  0.8888888955116272
<Relu>
Epoch 00117: early stopping
시간 :  12.06 초
1/1 [==============================] - 0s 123ms/step - loss: 0.1231 - accuracy: 0.9667
loss :  0.12309200316667557
accuracy :  0.9666666388511658

MinMaxScaler
Epoch 00408: early stopping
시간 :  46.06 초
2/2 [==============================] - 0s 974us/step - loss: 0.1322 - accuracy: 0.9444
loss :  0.13221938908100128
accuracy :  0.9444444179534912
<Relu>
Epoch 00153: early stopping
시간 :  15.41 초
1/1 [==============================] - 0s 157ms/step - loss: 0.0300 - accuracy: 1.0000
loss :  0.029968351125717163
accuracy :  1.0


StandardScaler
Epoch 00225: early stopping
시간 :  25.25 초
2/2 [==============================] - 0s 997us/step - loss: 0.1937 - accuracy: 0.9444
loss :  0.19366972148418427
accuracy :  0.9444444179534912
<Relu>
Epoch 00190: early stopping
시간 :  18.52 초
1/1 [==============================] - 0s 122ms/step - loss: 0.1492 - accuracy: 0.9667
loss :  0.1491539031267166
accuracy :  0.9666666388511658


RobustScaler
Epoch 00308: early stopping
시간 :  33.81 초
2/2 [==============================] - 0s 969us/step - loss: 0.1779 - accuracy: 0.9444
loss :  0.17789506912231445
accuracy :  0.9444444179534912
<Relu>
Epoch 00125: early stopping
시간 :  12.52 초
1/1 [==============================] - 0s 119ms/step - loss: 0.2126 - accuracy: 0.9667
loss :  0.2126045674085617
accuracy :  0.9666666388511658


MaxAbsScaler
Epoch 00154: early stopping
시간 :  17.33 초
2/2 [==============================] - 0s 997us/step - loss: 0.1858 - accuracy: 0.9167
loss :  0.18576472997665405
accuracy :  0.9166666865348816
epochs : 10000
<Relu>
Epoch 00141: early stopping
시간 :  14.26 초
1/1 [==============================] - 0s 121ms/step - loss: 0.1332 - accuracy: 0.9667
loss :  0.13317769765853882
accuracy :  0.9666666388511658







'''