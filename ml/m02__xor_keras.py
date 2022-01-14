import numpy as np
# from sklearn.svm import LinearSVC
# from sklearn.linear_model import Perceptron
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
#1. 데이터
x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [0,1,1,0]

#2. 모델
# model = LinearSVC()
# model = Perceptron()
model = Sequential()
model.add(Dense(1, input_dim =2, activation='sigmoid'))

#3. 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data,batch_size=1, epochs=100)

#4. 평가 예측
y_predict = model.predict(x_data)

result = model.evaluate(x_data,y_data)
print("model_acc : ",result[1])
print(x_data,"의 예측 결과 : ",y_predict)

# result = model.score(x_data,y_data)
# acc = accuracy_score(y_data, y_predict)
# print("model.score : ",result)
# print("accuracy_score : ",acc)
'''
[[0, 0], [0, 1], [1, 0], [1, 1]] 의 예측 결과 :  [[0.45932215]
model_acc :  0.75
'''