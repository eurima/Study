import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import optimizers, initializers, regularizers, metrics
import os
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

#1.데이터
datagen = ImageDataGenerator(
    rescale=1./255,             
    horizontal_flip=True,       
    vertical_flip=True,         
    width_shift_range=0.1,      
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'         
)        

xy  = datagen.flow_from_directory(
    'D:\_data\image\men_women',
    target_size=(150,150),
    batch_size=5,
    class_mode='categorical',
    shuffle=True,  
)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(xy[0][0], xy[0][1],
         train_size = 0.8, shuffle = True, random_state = 66)
#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout,MaxPooling2D

model = Sequential()
model.add(Conv2D(240,(2,2),activation = 'relu', input_shape =(150,150,3)))
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

model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=0.0001),metrics=['acc']) 
es = EarlyStopping(monitor='val_loss', patience=50, mode='auto', restore_best_weights=True)
# hiss = model.fit_generator(x_train,y_train, epochs=1000, 
#                     validation_split=0.2,
#                     callbacks=[es, mcp]
#                     ) 

hiss = model.fit(x_train,y_train, epochs=1000,validation_split=0.2,
                    callbacks=[es, mcp])

acc = hiss.history['acc']
val_acc = hiss.history['val_acc']
loss = hiss.history['loss']
val_loss = hiss.history['val_loss']

print('loss : ',loss[-1])
print('val_loss : ',val_loss[-1])
print('acc : ',acc[-1])
print('val_acc : ',val_acc[-1])

import matplotlib.pyplot as plt
x_len = np.arange(len(loss))
plt.plot(x_len, acc, marker='.', c="red", label='Trainset_acc')
plt.plot(x_len, val_acc, marker='.', c="lightcoral", label='Testset_acc')
plt.plot(x_len, val_loss, marker='.', c="cornflowerblue", label='Testset_loss')
plt.plot(x_len, loss, marker='.', c="blue", label='Trainset_loss')
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss/acc')
plt.show()
'''
loss :  0.0005020094686187804
val_loss :  0.0
acc :  1.0
val_acc :  1.0

'''
