from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#----1. 데이터 정제하여 투입
x = np.array([1,2,3,5,4])
y = np.array([1,2,3,4,5])

#----2. 모델 구현
model = Sequential() #Sequential 클래스의 인스턴스
model.add(Dense(1, input_dim = 1)) #출력 1, 인풋1 ( 1단 구조 )

#----3. 컴파일 , 훈련
model.compile(loss='mse', optimizer='adam') 
# mse : 최소자승법, adam 역할은 loss의 mse 감소 시키는 역할....일단은
epoch = 20000
model.fit(x, y, epochs = epoch, batch_size=1) 
#epochs 훈련양, batch_size 몇개씩 넣을 것인가 --> 속도, 성능, 과적합 여부에 따라

#----4. 성과 예측
loss = model.evaluate(x,y)
print('epochs : ',epoch)
print('loss : ', loss)
pre_x = 6
result = model.predict([pre_x])
print(f'{pre_x} 의 예측값은 : ',result[0][0])

'''
epochs :  1400
loss :  0.38376718759536743

epochs :  1500
loss :  0.3804122507572174

epochs :  1450
loss :  0.6376967430114746

epochs :  1600
loss :  0.39387741684913635

epochs :  1700
loss :  0.39398008584976196

epochs :  1800
loss :  0.38555318117141724

epochs :  3000
loss :  0.3861289620399475

epochs :  2500
loss :  0.386496365070343

epochs :  4000
loss :  0.3800128102302551

epochs :  8000
loss :  0.38000375032424927

epochs :  20000
loss :  0.38000187277793884    <-----------------

epochs :  200000
loss :  0.3800111711025238
'''