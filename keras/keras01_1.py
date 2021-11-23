# import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#----1. 데이터 정제하여 투입
x = np.array([1,2,3])
y = np.array([1,2,3])

#----2. 모델 구현
model = Sequential() #Sequential 클래스의 인스턴스
model.add(Dense(1, input_dim = 1)) #출력 1, 인풋1 ( 1단 구조 )

#----3. 컴파일 , 훈련
model.compile(loss='mse', optimizer='adam') 
# mse : 최소자승법, adam 역할은 loss의 mse 감소 시키는 역할....일단은
model.fit(x, y, epochs=2000, batch_size=1) 
#epochs 훈련양, batch_size 몇개씩 넣을 것인가 --> 속도, 성능, 과적합 여부에 따라

#----4. 성과 예측
loss = model.evaluate(x,y)
print('loss : ', loss)
pre_x = 4
result = model.predict([pre_x])
print(f'{pre_x} 의 예측값은 : ',result[0][0])
'''
2000  epoch
loss :  0.2507708966732025
4 의 예측값은 :  2.9459848
'''#Git 테스트

