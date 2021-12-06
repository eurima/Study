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
model.summary()

model.save_weights("./_save/keras25_1_save_weights.h5")

#3. 컴파일, 훈련
epoch = 10
model.compile(loss = 'mse', optimizer = 'adam')
start = time.time()
hist = model.fit(x_train, y_train, epochs = epoch, batch_size =1,validation_split=0.2)

end = time.time() - start
print('시간 : ', round(end,3) ,'초')

model.save_weights("./_save/keras25_3_save_weights.h5")

#4 평가예측
# loss = model.evaluate(x_test,y_test)
# print("loss : ",loss)

# y_predict = model.predict(x_test)
# r2 = r2_score(y_test,y_predict)
# print("R2 : ",r2)
# print(node_num)
# print("epochs :",epoch)
