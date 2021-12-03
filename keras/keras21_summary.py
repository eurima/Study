from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#----1. 데이터 정제하여 투입
x = np.array([1,2,3])
y = np.array([1,2,3])

#----2. 모델 구현
no = [5,3,4,2]
model = Sequential() #Sequential 클래스의 인스턴스
model.add(Dense(no[0], input_dim = 1)) 
model.add(Dense(no[1],activation = 'relu'))
model.add(Dense(no[2])) 
model.add(Dense(no[3])) 
model.add(Dense(1)) 

model.summary() #======> 전이학습시 유용하다 노드 구조파악
'''
#----3. 컴파일 , 훈련
model.compile(loss='mse', optimizer='adam') 
# mse : 최소자승법, adam 역할은 loss의 mse 감소 시키는 역할....일단은
epoch = 30
model.fit(x, y, epochs = epoch, batch_size=1) 
#epochs 훈련양, batch_size 몇개씩 넣을 것인가 --> 속도, 성능, 과적합 여부에 따라

#----4. 성과 예측
loss = model.evaluate(x,y)
print('epochs 고정 : ',epoch)
print('히든 레이어 구조 :',no)
print('loss : ', loss)
pre_x = 4
result = model.predict([pre_x])
print(f'{pre_x} 의 예측값은 : ',result[0][0])

'''
