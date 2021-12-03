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
scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = MaxAbsScaler()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)



#2 모델구성
#        
deep_len = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 40, 30, 20, 10, 5, 2]
model = Sequential()
model.add(Dense(deep_len[0], activation = 'linear', input_dim =x.shape[1]))
model.add(Dense(deep_len[1], )) # ===> 디폴트 값은 linear이고 sigmoid를 넣을 수도 있다 (값이 튀다면 sigmoid로 한번씩 잡아주면 성능이 좋아질 수 있다)
model.add(Dense(deep_len[2]))
model.add(Dense(deep_len[3],activation ='relu')) 
model.add(Dense(deep_len[4])) 
model.add(Dense(deep_len[5],activation ='relu'))
model.add(Dense(deep_len[6])) 
model.add(Dense(deep_len[7],activation ='relu')) 
model.add(Dense(deep_len[8])) 
model.add(Dense(deep_len[9])) 
# model.add(Dense(deep_len[10],activation ='relu'))
# model.add(Dense(deep_len[11])) 
# model.add(Dense(deep_len[12])) 
# model.add(Dense(deep_len[13],activation ='relu')) 
# model.add(Dense(deep_len[14])) 
# model.add(Dense(deep_len[15])) 
model.add(Dense(y.shape[1])) #이진분류의 마지막 레이어는 무조건 sigmoid!!!!
# sigmoid는 0 ~ 1 사이의 값을 뱉는다

#3. 컴파일, 훈련
epoch = 10000
model.compile(loss = 'mse', optimizer = 'adam', metrics=['accuracy']) # metrics=['accuracy'] 영향을 미치지 않는다

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

# print(test_flie)

scaler.transform(test_flie)
# ############### 제출용.
result = model.predict(test_flie)
submission['quality'] = result

# # print(submission[:10])
submission.to_csv(path+"sampleHR_StandardScaler.csv", index = False)

'''
Normal


MinMaxScaler



StandardScaler
Epoch 00051: early stopping
시간 :  93.96 초
21/21 [==============================] - 0s 847us/step - loss: 2.6905 - accuracy: 0.1437
loss :  2.6905014514923096
accuracy :  0.14374034106731415



RobustScaler
Epoch 00051: early stopping
시간 :  95.07 초
21/21 [==============================] - 0s 799us/step - loss: 12.7051 - accuracy: 0.0232
loss :  12.705144882202148
accuracy :  0.023183925077319145


MaxAbsScaler
Epoch 00051: early stopping
시간 :  90.08 초
21/21 [==============================] - 0s 787us/step - loss: 13.4276 - accuracy: 0.0263
loss :  13.427594184875488
accuracy :  0.026275115087628365







'''