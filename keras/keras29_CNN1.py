from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.python.keras.layers.core import Dropout


'''
model.summary()
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 4, 4, 10)          50
=================================================================
Total params: 50
Trainable params: 50
Non-trainable params: 0
'''
'''
model.add(Conv2D(10, kernel_size=(2,2), input_shape = (10,10,1) #9,9,10 아웃
model.add(Conv2D(5,(3,3), activation='relu')) # 7,7,5 아웃
model.add(Conv2D(7,(2,2), activation='relu')) # 6,6,7 아웃
in - kernel_size + 1

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 4, 4, 10)          50
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 3, 3, 5)           205
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 2, 2, 7)           147
=================================================================
Total params: 402
Trainable params: 402
왜 402인가?
(Conv2D(10, kernel_size=(2,2), input_shape = (10,10,1)
정식 명칭 찾아서 적을 것
'''
model = Sequential()
model.add(Conv2D(10, kernel_size=(2,2), input_shape = (10,10,1)))# 9,9,10
model.add(Conv2D(5,(3,3), activation='relu')) # 7,7,5
model.add(Dropout(0.2))
model.add(Conv2D(7,(2,2), activation='relu')) # 6,6,7 아웃  
# Dence 레이어에 입력 전에는 이차원으로 변형 해 줘야 한다->6*6*7 = 252
model.add(Flatten())
#flatten (Flatten)            (None, 252)   
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dense(5, activation='softmax'))
# model.summary()
'''
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 9, 9, 10)          50
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 7, 7, 5)           455
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 6, 6, 7)           147
_________________________________________________________________
flatten (Flatten)            (None, 252)               0
=================================================================
Total params: 652
Trainable params: 652

'''
