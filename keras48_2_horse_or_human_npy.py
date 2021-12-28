import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import optimizers, initializers, regularizers, metrics
import os


# #1.데이터 저장
# tr_datagen = ImageDataGenerator(
#     rescale=1./255,
# )
# xy = tr_datagen.flow_from_directory(
#     'D:\\_data\\image\\horse-or-human\\horse-or-human\\',
#     # '../_data/image/horse-or-human/horse-or-human/horses/',
#     target_size=(150,150),
#     batch_size=2000000, # 일단 크게 잡는다
#     class_mode='categorical',
# )
# x = np.save('D:\\Study\\_save_npy\\horse_x', arr=xy[0][0])
# y = np.save('D:\\Study\\_save_npy\\horse_y', arr=xy[0][1])
#
# print("完! numpy save complete~!")
# #2.데이터 불러오기

x = np.load('D:\\Study\\_save_npy\\horse_x.npy')
y = np.load('D:\\Study\\_save_npy\\horse_y.npy')

print(y.shape) #(1027, 2)
#
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x, y,
#          train_size = 0.8, shuffle = True, random_state = 66)
#
# ################### 폴더 관리 ###########################################
# file_name = os.path.abspath(__file__)
# filepath = "./_ModelCheckPoint/"
# dir_name = filepath + file_name.split("\\")[-1].split('.')[0]
# os.makedirs(dir_name, exist_ok=True)
#
# filepath = dir_name  # "./_ModelCheckPoint/"
# filename = '{epoch:04d}-{val_loss:4f}.hdf5'
# model_path = "".join([file_name.split("\\")[-1].split('.')[0], "_", filename])
# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True,
#                       filepath=filepath + "\\" + model_path)
#
# #2. 모델
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout,MaxPooling2D
#
# model = Sequential()
# model.add(Conv2D(32,(2,2),activation = 'relu', input_shape =(150,150,3)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(32, (3, 3),activation = 'relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(64, (3, 3),activation = 'relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(64,activation = 'relu'))
# model.add(Dropout(0.5))
# model.add(Dense(2,activation = 'softmax'))
#
# #3.컴파일, 훈련
#
# model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])
# es = EarlyStopping(monitor='val_loss', patience=50, mode='auto', restore_best_weights=True)
# hiss = model.fit(x_train,y_train, epochs=1000, validation_split=0.2, callbacks=[es, mcp] )
#
# acc = hiss.history['acc']
# val_acc = hiss.history['val_acc']
# loss = hiss.history['loss']
# val_loss = hiss.history['val_loss']
#
# print('loss : ',loss[-1])
# print('val_loss : ',val_loss[-1])
# print('acc : ',acc[-1])
# print('val_acc : ',val_acc[-1])

'''
al_acc: 1.0000
loss :  0.00013078250049147755
val_loss :  0.0003215078904759139
acc :  1.0
val_acc :  1.0

'''
