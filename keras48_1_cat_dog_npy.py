#세이브해서 로드
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import optimizers, initializers, regularizers, metrics
import os

'''
#1.데이터 저장
train_datagen = ImageDataGenerator(
    rescale=1./255,
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
    rescale=1./255              
)
xy_train  = train_datagen.flow_from_directory(
    '../_data/image/cat_dog/training_set/training_set/',
    target_size=(150,150),
    batch_size=2000, # 일단 크게 잡는다
    class_mode='categorical',
    shuffle=True,  
)
xy_test = test_datagen.flow_from_directory(
    '../_data/image/cat_dog/test_set/test_set/',
    target_size=(150,150),
    batch_size=2000,
    class_mode='categorical',
)
print(xy_train[0][0].shape, xy_train[0][1].shape)
print(xy_test[0][0].shape, xy_test[0][1].shape)

np.save('./_save_npy/cat_dog_train_x.npy', arr=xy_train[0][0])
np.save('./_save_npy/cat_dog_train_y.npy', arr=xy_train[0][1])
np.save('./_save_npy/cat_dog_test_x.npy', arr=xy_test[0][0])
np.save('./_save_npy/cat_dog_test_y.npy', arr=xy_test[0][1])

print("完! numpy save complete~!")
'''


x_train = np.load('D:\\Study\\_save_npy\\cat_dog_train_x.npy')
y_train = np.load('D:\\Study\\_save_npy\\cat_dog_train_y.npy')
x_test = np.load('D:\\Study\\_save_npy\\cat_dog_test_x.npy')
y_test = np.load('D:\\Study\\_save_npy\\cat_dog_test_y.npy')

#print(x_train.shape) #160,150,150,3

################### 폴더 관리 ###########################################
file_name = os.path.abspath(__file__)
filepath = "./_ModelCheckPoint/"
dir_name = filepath + file_name.split("\\")[-1].split('.')[0]
os.makedirs(dir_name, exist_ok=True)

filepath = dir_name  # "./_ModelCheckPoint/"
filename = '{epoch:04d}-{val_loss:4f}.hdf5'
model_path = "".join([file_name.split("\\")[-1].split('.')[0], "_", filename])
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True,
                      filepath=filepath + "\\" + model_path)

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout,MaxPooling2D

model = Sequential()
model.add(Conv2D(32,(2,2),activation = 'relu', input_shape =(150,150,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3),activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3),activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64,activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(2,activation = 'softmax'))

#3.컴파일, 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc']) 
es = EarlyStopping(monitor='val_loss', patience=50, mode='auto', restore_best_weights=True)
hiss = model.fit(x_train,y_train, epochs=1000, validation_split=0.2, callbacks=[es, mcp] )


acc = hiss.history['acc']
val_acc = hiss.history['val_acc']
loss = hiss.history['loss']
val_loss = hiss.history['val_loss']

print('loss : ',loss[-1])
print('val_loss : ',val_loss[-1])
print('acc : ',acc[-1])
print('val_acc : ',val_acc[-1])

'''
loss :  0.045295003801584244
val_loss :  2.5832135677337646
acc :  0.9831249713897705
val_acc :  0.6050000190734863

'''


