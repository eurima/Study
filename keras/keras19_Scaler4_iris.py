from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import time

dataset = load_iris()

x = dataset.data
y = dataset.target

from tensorflow.keras.utils import to_categorical
# one_hot = to_categorical(y,num_classes=len(np.unique(y)))
y = to_categorical(y) #<=============== class 개수대로 자동으로 분류 해 준다!!! /// 간단!!

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
         train_size = 0.8, shuffle = True, random_state = 66) #455.2 /114

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = RobustScaler()
scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



#2 모델구성
#
deep_len = [100, 50, 30, 20, 100, 50, 30, 40, 50, 40, 30, 20, 10, 5, 4, 2]
model = Sequential()
model.add(Dense(deep_len[0], activation = 'linear', input_dim =x.shape[1]))
model.add(Dense(deep_len[1], )) # ===> 디폴트 값은 linear이고 sigmoid를 넣을 수도 있다 (값이 튀다면 sigmoid로 한번씩 잡아주면 성능이 좋아질 수 있다)
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
model.add(Dense(y.shape[1], activation = 'softmax')) #이진분류의 마지막 레이어는 무조건 sigmoid!!!!
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
#===========> 가장 중요한것은 Loss 이다!!
#===========> Loss 가 가장 낮은 모델이 무조건 좋은 것이다!!!
#===========> 더욱 중요한것은 val_loss 이다!!
y_predict = model.predict(x_test)
print("epochs :",epoch)

'''
Normal
Epoch 00315: early stopping
시간 :  33.48 초
1/1 [==============================] - 0s 165ms/step - loss: 0.0653 - accuracy: 0.9667
loss :  0.06529214233160019
accuracy :  0.9666666388511658

MinMaxScaler
Epoch 00128: early stopping
시간 :  12.79 초
1/1 [==============================] - 0s 124ms/step - loss: 0.0723 - accuracy: 0.9667
loss :  0.07231131196022034
accuracy :  0.9666666388511658

StandardScaler
Epoch 00087: early stopping
시간 :  8.95 초
1/1 [==============================] - 0s 120ms/step - loss: 0.0627 - accuracy: 0.9667
loss :  0.06272416561841965
accuracy :  0.9666666388511658

RobustScaler
Epoch 00133: early stopping
시간 :  13.14 초
1/1 [==============================] - 0s 117ms/step - loss: 0.0841 - accuracy: 0.9667
loss :  0.08412198722362518
accuracy :  0.9666666388511658

MaxAbsScaler
Epoch 00215: early stopping
시간 :  20.51 초
1/1 [==============================] - 0s 120ms/step - loss: 0.0657 - accuracy: 0.9667
loss :  0.0656883716583252
accuracy :  0.9666666388511658




'''