import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

# print('훈련용 리뷰 개수 : {}'.format(len(x_train)))
# print('테스트용 리뷰 개수 : {}'.format(len(x_test)))
# num_classes = len(set(y_train))
# print('카테고리 : {}'.format(num_classes))
# print(x_train.shape,y_train.shape) #(25000,) (25000,)
# print(np.unique(y_train))#0,1[0 1]
# print(x_train[0])
# print(y_train[0])#1
# x_train_len = max(len(i) for i in x_train)    #2494
# x_train_avg = sum(map(len,x_train)/len(x_train)) #평균길이 143.53
# print(x_train_len)
# print(x_train_avg)
max_len = 500
vocab_size = 10000

## 전처리
from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(x_train,padding='pre',maxlen=max_len, truncating='pre')
x_test = pad_sequences(x_test,padding='pre',maxlen=max_len, truncating='pre') 

#2모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM, Embedding, Flatten,Dropout

model = Sequential()
model.add(Embedding(vocab_size, 128))
model.add(LSTM(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('reuter_best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(x_train, y_train, validation_split=0.2, batch_size=32, epochs=3000, callbacks=[es, mc])

print("\n 테스트 정확도: %.4f" % (model.evaluate(x_test, y_test)[1]))
print("\n 테스트 Loss: %.4f" % (model.evaluate(x_test, y_test)[0]))
y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)

print("R2 : ",r2)

epochs = range(1, len(history.history['acc']) + 1)
plt.plot(epochs, history.history['loss'])
plt.plot(epochs, history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
