import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#1.데이터
train_datagen = ImageDataGenerator(
    rescale=1./255,             #MinMaxScaler
    horizontal_flip=True,       #상하반전 이미지도 동일한 라벨
    vertical_flip=True,         #좌우반전 이미지도 동일한 라벨
    width_shift_range=0.1,      #좌우 이동시도 동일한 이미지로 인식
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'         #이미지 이동시 빈 공백을 주변의 비슷한 값으로 채우겠다
)
test_datagen = ImageDataGenerator(
    rescale=1./255              #test파일은 스케일만 해 준다
)
#D:\_data\image\brain
xy_train  = train_datagen.flow_from_directory(
    '../_data/image/brain/train/',
    target_size=(150,150),
    batch_size=5,
    class_mode='binary',
#categorical : 2D one-hot 부호화된 라벨이 반환됩니다.
#binary : 1D 이진 라벨이 반환됩니다.
#sparse : 1D 정수 라벨이 반환됩니다.
#None : 라벨이 반환되지 않습니다.
    shuffle=True,  
)
#Found 160 images belonging to 2 classes.
xy_test = test_datagen.flow_from_directory(
    '../_data/image/brain/test/',
    target_size=(150,150),
    batch_size=5,
    class_mode='binary',
)
#Found 120 images belonging to 2 classes.
# print(xy_train[0]) #----> x,y가 뭉쳐진 데이터가 1 batch나옴 
# 60개의 데이터,batch_siz=5 이므로 [32]까지 나옴
# print(xy_train[0][0]) # x 값
# print(xy_train[0][1]) # y 값

# print(xy_train[0][0].shape) #(5,150,150,3) batch_size
# print(xy_train[0][1].shape) #(5,)

# print(type(xy_train)) #<class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
# print(type(xy_train[0])) #<class 'tuple'>
# print(type(xy_train[0][0])) #<class 'numpy.ndarray'>
# print(type(xy_train[0][1])) #<class 'numpy.ndarray'>