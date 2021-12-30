from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import reuters
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
#1데이터
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=10000, test_split=0.2)
# print('훈련용 뉴스 기사 : {}'.format(len(X_train)))
# print('테스트용 뉴스 기사 : {}'.format(len(X_test)))
# num_classes = len(set(y_train))
# print('카테고리 : {}'.format(num_classes))
# print(x_train.shape,y_train.shape) #(8982,) (8982,)
# x_train_len = 0
# for list_x in x_train:
#     if x_train_len < len(list_x):
#         x_train_len = len(list_x)
# print(x_train_len) #2376
# x_train_len = max(len(i) for i in x_train)    #2376
# x_train_avg = sum(map(len,x_train)/len(x_train)) #평균길이 143.53

## 전처리
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# x_train = pad_sequences(x_train,padding='pre') #8982개수,2376최대길이
max_len = 100
x_train = pad_sequences(x_train,padding='pre',maxlen=max_len, truncating='pre') 
#truncating='pre'-> (8982,100) 필요없는 앞부분 자른다
x_test = pad_sequences(x_test,padding='pre',maxlen=max_len, truncating='pre') 

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# print(x_train.shape, y_train.shape)#(8982, 100) (8982, 46)

#2모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM, Embedding, Flatten,Dropout

model = Sequential()
model.add(Embedding(1000, 128))
model.add(LSTM(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(46, activation='softmax'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('reuter_best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(x_train, y_train, validation_split=0.2, batch_size=32, epochs=3000, callbacks=[es, mc])

print("\n 테스트 정확도: %.4f" % (model.evaluate(x_test, y_test)[1]))
print("\n 테스트 Loss: %.4f" % (model.evaluate(x_test, y_test)[0]))

epochs = range(1, len(history.history['acc']) + 1)
plt.plot(epochs, history.history['loss'])
plt.plot(epochs, history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

