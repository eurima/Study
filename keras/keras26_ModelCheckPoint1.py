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

node_num = [40,30,20,10]
model = Sequential() 
model.add(Dense(node_num[0], input_dim = 13)) 
model.add(Dense(node_num[1])) 
model.add(Dense(node_num[2]))
model.add(Dense(node_num[3])) 
model.add(Dense(1)) 
# model.summary()

# model.save_weights("./_save/keras25_1_save_weights.h5") <----- 훈련 전 weight를 저장(랜덤값)



#3. 컴파일, 훈련
epoch = 100
patience_num = 10
model.compile(loss ='mse', optimizer = 'adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=patience_num, mode = 'auto', restore_best_weights=False)
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose=1, save_best_only= False, filepath= "./_ModelCheckPoint/keras26_1_MCP.hdf5")
# -> verbose 옵션으로 val_loss improved from 73.58737 to 65.55513, saving model to ./_ModelCheckPoint\keras26_1_MCP.hdf5 이렇게 표시됨
start = time.time()
hist = model.fit(x_train, y_train, epochs = epoch, batch_size =1,validation_split=0.2, callbacks=[es, mcp])
end = time.time() - start

print('걸린 시간 : ', round(end,3) ,'초')

# model.load_weights("./_save/keras25_3_save_weights.h5") #<----- 통상적으로 컴파일 다음에 넣는다
model.save_weights("./_save/keras26_1_save_weights.h5")

#4 평가예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print("R2 : ",r2)
# print(node_num)
# print("epochs :",epoch)

print('*******************')
print('hist:',hist)
print('*******************')
print('history:',hist.history)
print('*******************')
print(hist.history['loss'])
print('*******************')
print(hist.history['val_loss'])

import matplotlib.pyplot as plt
plt.figure(figsize=(9,5))
plt.plot(hist.history['loss'],marker = '.',c='red',label = 'loss')
plt.plot(hist.history['val_loss'],marker = '.',c='blue',label = 'val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')
plt.show()

'''
loss :  32.87297439575195
R2 :  0.6067023569508212

'''

