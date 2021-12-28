import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import optimizers, initializers, regularizers, metrics
import os


#1.데이터 저장
# tr_datagen = ImageDataGenerator(
#     rescale=1./255,
#     horizontal_flip=True,       #상하반전 이미지도 동일한 라벨
#     vertical_flip=True,         #좌우반전 이미지도 동일한 라벨
#     width_shift_range=0.1,      #좌우 이동시도 동일한 이미지로 인식
#     height_shift_range=0.1,
#     rotation_range=5,
#     zoom_range=1.2,
#     shear_range=0.7,
#     fill_mode='nearest'         #이미지 이동시 빈 공백을 주변의 비슷한 값으로 채우겠다    
# )
# xy = tr_datagen.flow_from_directory(
#     'D:\\_data\\image\\men_women\\',
#     target_size=(150,150),
#     batch_size=2000000, # 일단 크게 잡는다
#     class_mode='categorical',
# )
# print(xy[0][0])
# print(xy[0][1])

# x = np.save('D:\\Study\\_save_npy\\men_women_x', arr=xy[0][0])
# y = np.save('D:\\Study\\_save_npy\\men_women_y', arr=xy[0][1])

# print("完! numpy save complete~!")
###2.데이터 불러오기

x = np.load('D:\\Study\\_save_npy\\men_women_x.npy')
y = np.load('D:\\Study\\_save_npy\\men_women_y.npy')

print(x.shape) #3309,150,150,3
print(y.shape) #3309,2


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
         train_size = 0.8, shuffle = True, random_state = 66)

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

######2. 모델
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

####3.컴파일, 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=50, mode='auto', restore_best_weights=True)
hiss = model.fit(x_train,y_train, epochs=100, validation_split=0.2, callbacks=[es, mcp] )

acc = hiss.history['acc']
val_acc = hiss.history['val_acc']
loss = hiss.history['loss']
val_loss = hiss.history['val_loss']

print('loss : ',loss[-1])
print('val_loss : ',val_loss[-1])
print('acc : ',acc[-1])
print('val_acc : ',val_acc[-1])

'''
loss :  0.04094623401761055
val_loss :  3.353677272796631
acc :  0.9877184629440308
val_acc :  0.6415094137191772

'''



