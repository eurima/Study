import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, GRU
from sklearn.preprocessing import StandardScaler
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
model.add(GRU(10,activation = 'linear', input_shape = (3,1)))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1))
model.summary()
'''
RNN 파라
params = dim(W)+dim(V)+dim(U) = n*n + kn + nm

# n - dimension of hidden layer
# k - dimension of output layer 
# m - dimension of input layer

model.add(SimpleRNN(12,activation = 'linear', input_shape = (1,3)))

Dh = 0
t = 1
d = 3

Dh * Dh + Dh * d + Dh
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
gru (GRU)                    (None, 10)                390
_________________________________________________________________
dense (Dense)                (None, 10)                110
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 11
=================================================================
Total params: 511
Trainable params: 511
'''

 