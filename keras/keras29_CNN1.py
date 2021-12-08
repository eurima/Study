from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.python.keras.layers.core import Dropout


'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 4, 4, 10)          50
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 3, 3, 5)           205
=================================================================
Total params: 255
Trainable params: 255
Non-trainable params: 0

왜 50인가?
model.add(Conv2D(10, kernel_size = (2,2), input_shape = (10,10, 1))  

model.add(Conv2D(출력채널, kernel_size = (필터크기) , input_shape = (,,RGB)))

(필터 크기 axb) x  (입력 채널(RGB)) x (출력 채널) + (출력 채널 bias)

model.add(Conv2D(10, kernel_size = (2,2), input_shape = (10,10,1)))

2 * 2 * 1 * 10 + 10 = 40 + 10 = 50

두번째 레이어는 입력 받는 채널이 10, 필터크기 (2,2) , 필터 5
2 * 2 * 10 * 5 + 5 = 205
=====================


(Conv2D(10, kernel_size=(2,2), input_shape = (10,10,1)
정식 명칭 찾아서 적을 것

첫번째 인자 : filter 컨볼루션 필터의 수
두번째 인자 : kernel_size 컨볼루션 커널의 (행,열)
세번째인자 샘플수를 제외한 입력 형태를 정의
모델에서의 첫 레이어일때만 정의
(행,열,채널수)
흑백의 경우 채널이 1, 컬러는 채널 3
행 = 너비 width
열 = 높이 heighr

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
model.add(Dropout(0.5))
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
