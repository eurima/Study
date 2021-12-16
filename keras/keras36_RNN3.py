import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
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

#2 모델구성
model = Sequential()
model.add(SimpleRNN(10, input_length=3, input_dim = 1)) 
#input_shape = (3,1) -> RNN 계열에서 이렇게 쓰기도
model.summary()