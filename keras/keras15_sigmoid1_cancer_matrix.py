from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from sklearn.metrics import r2_score
import time

#1 데이터
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
model = Sequential()
model.add(Dense(deep_len[0], activation = 'linear', input_dim = 30)) 
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
model.add(Dense(1, activation = 'sigmoid')) #이진분류의 마지막 레이어는 무조건 sigmoid!!!!
# sigmoid는 0 ~ 1 사이의 값을 뱉는다

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

model.fit(x_train, y_train, epochs = epoch, batch_size =1,validation_split=0.2,callbacks=[es])
end = time.time() - start
print('시간 : ', round(end,2) ,'초')

#4 평가예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss)

#===========> 가장 중요한것은 Loss 이다!!
#===========> Loss 가 가장 낮은 모델이 무조건 좋은 것이다!!!
#===========> 더욱 중요한것은 val_loss 이다!!


y_predict = model.predict(x_test)
# r2 = r2_score(y_test,y_predict)  ============ 이진분류에서 R2는 사용하지 않는다
# print("R2 : ",r2)
# print(deep_len)
print("epochs :",epoch)

def binary_print(num):
    if num > 0.5:
        return 1
    else:
        return 0

print(y_test[:10])

print(binary_print(y_predict[:10]))

'''
시그모이드 정리 해 주세요
'''

'''
Epoch 00157: early stopping
시간 :  46.74 초
4/4 [==============================] - 0s 998us/step - loss: 0.3355
loss :  0.33549413084983826
R2 :  0.7074553996457688
[100, 40, 30, 20, 10, 20, 30, 40, 50, 40, 30, 20, 10, 5, 4, 2]

364/364 [==============================] - 0s 746us/step - loss: 0.1414 - val_loss: 0.0723 
Epoch 00186: early stopping
시간 :  54.8 초
4/4 [==============================] - 0s 0s/step - loss: 0.2093
loss :  0.20928820967674255
R2 :  0.7195918532172652
[100, 50, 30, 20, 10, 20, 30, 40, 50, 40, 30, 20, 10, 5, 4, 2]

364/364 [==============================] - 0s 811us/step - loss: 0.1495 - val_loss: 0.1447
Epoch 00176: early stopping
시간 :  54.3 초
4/4 [==============================] - 0s 0s/step - loss: 0.2054
loss :  0.2053930163383484
R2 :  0.7251985411307253
[100, 80, 60, 40, 50, 40, 30, 20, 10, 50, 40, 30, 20, 10, 5, 2]
------------------
sigmoid를 사용한경우
364/364 [==============================] - 0s 895us/step - loss: 0.6630 - val_loss: 0.6609
Epoch 00182: early stopping
시간 :  54.87 초
4/4 [==============================] - 0s 665us/step - loss: 0.6539
loss :  0.6539096236228943
R2 :  -0.0014065813455694798
[100, 50, 30, 20, 10, 20, 30, 40, 50, 40, 30, 20, 10, 5, 4, 2]

364/364 [==============================] - 0s 894us/step - loss: 0.2982 - val_loss: 0.2661
Epoch 00264: early stopping
시간 :  89.85 초
4/4 [==============================] - 0s 0s/step - loss: 0.3636
loss :  0.36363959312438965
R2 :  0.4937792331366312
[200, 180, 160, 140, 150, 140, 130, 120, 100, 50, 40, 30, 20, 10, 5, 2]
============================================================================================================
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
364/364 [==============================] - 0s 839us/step - loss: 0.1428 - accuracy: 0.9313 - val_loss: 0.0719 - val_accuracy: 0.9780
Restoring model weights from the end of the best epoch.
Epoch 00236: early stopping
시간 :  73.37 초
4/4 [==============================] - 0s 5ms/step - loss: 0.1827 - accuracy: 0.9298
loss :  [0.1826707422733307, 0.9298245906829834]

364/364 [==============================] - 0s 845us/step - loss: 0.1432 - accuracy: 0.9368 - val_loss: 0.0679 - val_accuracy: 0.9670
Restoring model weights from the end of the best epoch.
Epoch 00230: early stopping
시간 :  73.33 초
4/4 [==============================] - 0s 0s/step - loss: 0.1877 - accuracy: 0.9386
loss :  [0.1876782774925232, 0.9385964870452881]
'''

