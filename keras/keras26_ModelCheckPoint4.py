from tensorflow.keras.models import Sequential, load_model
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

#3. 컴파일, 훈련
epoch = 10000
patience_num = 100
model.compile(loss ='mse', optimizer = 'adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
########################################################################
import datetime
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")
filepath = "./_ModelCheckPoint/"
filename = '{epoch:04d}-{val_loss:4f}.hdf5' #filepath + datetime
model_path = "".join([filepath,'k26_',datetime,"_",filename])
########################################################################
es = EarlyStopping(monitor='val_loss', patience=patience_num, mode = 'auto', restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose=1, save_best_only= True, filepath = model_path)
# -> verbose 옵션으로 val_loss improved from 73.58737 to 65.55513, saving model to ./_ModelCheckPoint\keras26_1_MCP.hdf5 이렇게 표시됨
start = time.time()
hist = model.fit(x_train, y_train, epochs = epoch, batch_size =1,validation_split=0.2, callbacks=[es, mcp])

end = time.time() - start

# print('걸린 시간 : ', round(end,3) ,'초')
# model.load_weights("./_save/keras25_3_save_weights.h5") #<----- 통상적으로 컴파일 다음에 넣는다
model.save("./_save/keras26_3_save_weights.h5")

#4 평가예측
############################################ 기본출력
print("====================   1 .  기 본 출 력     ==========================")
loss = model.evaluate(x_test,y_test)
print("loss : ",loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print("R2 : ",r2)
# print(node_num)

print("====================   2 .  Load_model 출력   =========================")
model = load_model("./_save/keras26_3_save_weights.h5")
loss = model.evaluate(x_test,y_test)
print("loss : ",loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print("R2 : ",r2)
# print(node_num)

# print("====================   3 .  Model Check Point 출력   ==================")
# model = load_model("./_ModelCheckPoint/keras26_3_MCP.hdf5")
# loss = model.evaluate(x_test,y_test)
# print("loss : ",loss)

# y_predict = model.predict(x_test)
# r2 = r2_score(y_test,y_predict)
# print("R2 : ",r2)



