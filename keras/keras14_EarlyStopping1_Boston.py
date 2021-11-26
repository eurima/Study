from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston
import time

#1.데이터
dataset = load_boston()
x = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.8, shuffle = True, random_state = 66) 


deep_len = [50,40,30,20,10,20,30,40,50,40,30,20,10,5,4,2]
model = Sequential() 
model.add(Dense(deep_len[0], input_dim = 13)) 
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
model.add(Dense(1)) 

#3. 컴파일, 훈련
epoch = 10000
model.compile(loss = 'mse', optimizer = 'adam')

from tensorflow.keras.callbacks import EarlyStopping
patience_num = 50
es = EarlyStopping(monitor='val_loss', patience=patience_num, mode = 'min', verbose=1)

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

# import matplotlib.pyplot as plt
# plt.figure(figsize=(9,5))
# plt.plot(hist.history['loss'],marker = '.',c='red',label = 'loss')
# plt.plot(hist.history['val_loss'],marker = '.',c='blue',label = 'val_loss')
# plt.grid()
# plt.title('loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(loc = 'upper right')
# plt.show()

print(hist.history['val_loss'])

'''
Epoch 00293: early stopping
R2 :  0.7768697433341409
val_loss: 33.0962

출력된 val_loss의 값은 모두 293 개 입니다
가장 낮은 val_loss는  32.24910354614258
293번째 val_loss는  33.0962028503418
가장 낮은 val_loss는 243 번째 epoch 입니다
Early Stop 구간의 val_loss가 가장 낮은 값은 아닙니다        
Early Stop 에서 patience 앞의 val_loss가 가장 낮은 값 입니다!
'''

'''
https://github.com/keras-team/keras/blob/v2.7.0/keras/callbacks.py#L1710-L1855

restore_best_weights 옵션:
True 인경우 -> Early Stop 된 구간에서이 가장 낮은 loss 의 Epoch 의 weight을 저장
False 인경우 -> 가장 최종의 Epoch 의 weight을 저장

restore_best_weights: Whether to restore model weights from
        the epoch with the best value of the monitored quantity.
        If False, the model weights obtained at the last step of
        training are used. An epoch will be restored regardless
        of the performance relative to the `baseline`. If no epoch
        improves on `baseline`, training will run for `patience`
        epochs and restore weights from the best epoch in that set.
'''