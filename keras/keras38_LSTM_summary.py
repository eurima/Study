import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.callbacks import EarlyStopping
#1 데이터
x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6]])
y = np.array([4,5,6,7])

print(x.shape, y.shape) #(4,3), (4,)

#input_shape = (batch_size, timestep, feature) (행, 열, 몇개씩 자르는지)
x = x.reshape(4,3,1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.9, shuffle = True, random_state = 66) 
scaler = StandardScaler()
# print(x)
#2 모델구성
model = Sequential()
# model.add(LSTM(10,activation = 'linear', input_shape = (3,1)))
model.add(SimpleRNN(10,activation = 'linear', input_shape = (3,1)))
model.add(Dense(10))
model.add(Dense(1))
model.summary()
'''
SimpleRNN
_________________________________________________________________
simple_rnn (SimpleRNN)       (None, 10)                120

LSTM
_________________________________________________________________
lstm (LSTM)                  (None, 10)                480

LSTM 연산 법 >>>>>>>>>>>
Dense 에 비해 4배의 연산수가 나오는 이유
LSTM has 4 dense layers in its internal structure 

i= input size
h= size of hidden layer (number of neurons in the hidden layer)
o= output size (number of neurons in the output layer)
g, no. of FFNNs in a unit (RNN has 1, GRU has 3, LSTM has 4)

g = 4 (LSTM has 4 FFNNs)
= g × [h(h+i) + h]

480 = 4 X[10*(10+1)+10) = 120*4

 
LSTM 스텝 (g의 값이 4인 이유)>>>>>>>>>>>>>>
1. Forget Gate layer 셀게이트에서 어떤 정보를 버릴지 선택하는 과정 
- Sigmoid 1: 유지 0: 버리다

새로운 정보가 셀스테이트에 저장될지 결정
2. Input Gate layer - 어떤 값을 업데이트 할지 결정
3. tanh layer - 셀 스테이트에 더해질 수 있는 새로운 후보 값을 만들어 낸다
두 가지 값을 합쳐서 다음 스테이트에 영향

4. 마지막으로 무엇을 생산할지 결정해야
Cell State의 어떤 부분을 출력할 것인지 결정하는 Sigmoid Layer를 실행
Cell State를 tanh를 통해 넣고 - 값을 -1과 1 사이로 밀어넣기 위해 -
Sigmoid Gate의 출력에 곱하기 때문에 결정한 부분만 출력

 Hyperbolic Tangent(tanh)>>>>>>>>>>>>
 Hyperbolic Tangent 함수는 확장 된 시그모이드 함수
 tanh와 Sigmoid의 차이점은 Sigmoid의 출력 범위가 0에서 1 사이인 반면 
 tanh와 출력 범위는 -1에서 1사이
 
 Sigmoid와 비교하여 tanh와는 출력 범위가 더 넓고 
 경사면이 큰 범위가 더 크기 때문에 더 빠르게 수렴하여 학습하는 특성
 - 중앙값이 0이기 때문에 경사하강법 사용 시 시그모이드 함수에서 발생하는 편향 이동이 발생하지 않는다
 - Sigmoid의 치명적인 단점인 Vanishing gradient problem 문제를 그대로 갖고 있다. 
 
 
 
'''
