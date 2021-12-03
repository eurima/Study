from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential , Model
from tensorflow.keras.layers import Dense, Input
# from sklearn.metrics import r2_score
import time

dataset  = load_breast_cancer()
# print(dataset)
# print(dataset.DESCR)
# print(dataset.feature_name)

x = dataset.data
y = dataset.target

# print(x.shape, y.shape) #(569,30) (569,)
# print(np.unique(y)) #----> [0, 1] : 배열의 고유값을 찾아준다 (라벨값이 어떤것이 있는가)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.8, shuffle = True, random_state = 66) #455.2 /114

#2 모델구성
#          [100, 50, 30, 20, 10, 20, 30, 40, 50, 40, 30, 20, 10, 5, 4, 2]
deep_len = [100, 50, 30, 20, 100, 50, 30, 40, 50, 40, 30, 20, 10, 5, 4, 2]
# model = Sequential()
# model.add(Dense(deep_len[0], activation = 'linear', input_dim = 30)) 
# model.add(Dense(deep_len[1], )) # ===> 디폴트 값은 linear이고 sigmoid를 넣을 수도 있다 (값이 튀다면 sigmoid로 한번씩 잡아주면 성능이 좋아질 수 있다)
# model.add(Dense(deep_len[2]))
# model.add(Dense(deep_len[3])) 
# model.add(Dense(deep_len[4])) 
# model.add(Dense(deep_len[5])) 
# model.add(Dense(deep_len[6])) 
# model.add(Dense(deep_len[7])) 
# model.add(Dense(deep_len[8])) 
# model.add(Dense(deep_len[9])) 
# model.add(Dense(deep_len[10])) 
# model.add(Dense(deep_len[11])) 
# model.add(Dense(deep_len[12])) 
# model.add(Dense(deep_len[13])) 
# model.add(Dense(deep_len[14])) 
# model.add(Dense(deep_len[15])) 
# model.add(Dense(1, activation = 'sigmoid')) #이진분류의 마지막 레이어는 무조건 sigmoid!!!!
# sigmoid는 0 ~ 1 사이의 값을 뱉는다
input1 = Input(shape=(x.shape[1],))

dense1 = Dense(deep_len[0])(input1)
dense2 = Dense(deep_len[1])(dense1)
dense3 = Dense(deep_len[2])(dense2)
dense4 = Dense(deep_len[3])(dense3)
dense5 = Dense(deep_len[4])(dense4)
dense6 = Dense(deep_len[5])(dense5)
dense7 = Dense(deep_len[6])(dense6)
dense8 = Dense(deep_len[7])(dense7)
dense9 = Dense(deep_len[8])(dense8)
dense10 = Dense(deep_len[9])(dense9)
dense11 = Dense(deep_len[10])(dense10)
dense12 = Dense(deep_len[11])(dense11)
dense13 = Dense(deep_len[12])(dense12)
dense14 = Dense(deep_len[13])(dense13)
dense15 = Dense(deep_len[14])(dense14)
dense16 = Dense(deep_len[15])(dense15)
output1 = Dense(1, activation = 'sigmoid')(dense16)

model = Model(inputs = input1, outputs = output1)


#3. 컴파일, 훈련
epoch = 10000
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy']) # metrics=['accuracy'] 영향을 미치지 않는다

from tensorflow.keras.callbacks import EarlyStopping
patience_num = 50
es = EarlyStopping(monitor='val_loss', patience=patience_num, mode = 'auto', verbose=1, restore_best_weights=True)
# 통상 val_loss 가 성능이 더 좋다
# 그렇지만 너무 튄다면 loss를 넣어도 된다
# 나중에는 monitor='accuracy' ,mode 가 헷갈리면 'auto'로 잡는다
start = time.time()

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = RobustScaler()
scaler = MaxAbsScaler()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)



model.fit(x_train, y_train, epochs = epoch, batch_size =1,validation_split=0.2,callbacks=[es])
end = time.time() - start
print('시간 : ', round(end,2) ,'초')

#4 평가예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss[0]) 
print("accuracy : ",loss[1])

y_predict = model.predict(x_test)
print("epochs :",epoch)

def binary_print(num):
    if num > 0.5:
        return 1
    else:
        return 0
    
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




'''
