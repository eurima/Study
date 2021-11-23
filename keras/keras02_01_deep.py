from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#----1. 데이터 정제하여 투입
x = np.array([1,2,3])
y = np.array([1,2,3])

#----2. 모델 구현
model = Sequential() #Sequential 클래스의 인스턴스
model.add(Dense(5, input_dim = 1)) #출력 5, 인풋1
model.add(Dense(3)) #출력 3 (인풋5) Sequential구조이기 때문에 인풋 별도 명시 X
model.add(Dense(4)) #출력 4 (인풋3)
model.add(Dense(2)) #출력 2 (인풋4)
model.add(Dense(1)) #출력 1 (인풋2)

#----3. 컴파일 , 훈련
model.compile(loss='mse', optimizer='adam') 
# mse : 최소자승법, adam 역할은 loss의 mse 감소 시키는 역할....일단은
epoch = 100
model.fit(x, y, epochs = epoch, batch_size=1) 
#epochs 훈련양, batch_size 몇개씩 넣을 것인가 --> 속도, 성능, 과적합 여부에 따라

#----4. 성과 예측
loss = model.evaluate(x,y)
print('epochs : ',epoch)
print('loss : ', loss)
pre_x = 4
result = model.predict([pre_x])
print(f'{pre_x} 의 예측값은 : ',result[0][0])

'''
epochs :  100
loss :  0.02614203654229641
4 의 예측값은 :  3.6633067

epochs :  200
loss :  0.01586858369410038
4 의 예측값은 :  3.7407138

epochs :  300
loss :  0.00024098118592519313
4 의 예측값은 :  3.9686794


'''
