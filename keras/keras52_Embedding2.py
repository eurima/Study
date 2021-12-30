from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

#1데이터
docs = ['너무 재밋어요', '참 최고예요','참 잘 만든 영화예요',
        '추천하고 싶은 영화입니다', '한 번 더 보고 싶네요', '글쎄요',
        '별로예요','생각보다 지루해요','연기가 어색해요',
        '재미없어요', '너무 재미없다','참 재밋네요','예람이가 잘 생기긴 했어요']

#긍정1, 부정0
lable = np.array([1,1,1,1,1,1,0,0,0,0,0,1,1])

token = Tokenizer()
token.fit_on_texts(docs)
x= token.texts_to_sequences(docs)
from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre', maxlen=5)
word_size = len(token.word_index)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM

#
model = Sequential()
model.add(Embedding(100, 10))
model.add(LSTM(32))
model.add(Dense(1,activation='sigmoid'))
# model.summary()

#3. 컴파일, 훈련
opt="adam"
model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics=['acc']) # metrics=['accuracy'] 영향을 미치지 않는다
model.fit(pad_x, lable, epochs = 100, batch_size=32)

########################################################################
#4 평가예측
from sklearn.metrics import r2_score, accuracy_score

acc = model.evaluate(pad_x,lable)[1]
print("acc : ",acc)
'''
acc :  1.0
'''
###실습 소스를 완성하여라 결과는 긍정? 부정?
x_predict = "나는 반장이 정말 재미 없다 정말"
print(x_predict)
# x_predict = x_predict.split(' ')
x_predict = token.texts_to_sequences([x_predict])
x_predict = pad_sequences(x_predict,padding='pre', maxlen=5)
print(model.predict(x_predict))
score = model.predict(x_predict)[0][0]

if (score > 0.5):
         print("{:.2f}% 확률로 긍정 입니다.\n".format(score * 100))
else:
         print("{:.2f}% 확률로 부정 입니다.\n".format((1 - score) * 100))


'''
나는 반장이 정말 정말 재미 없다
63.77% 확률로 부정 입니다.

0.37794816
62.21% 확률로 부정 입니다.

'''


'''
# x_predict = "나는 재미 있다 "
x_predict = x_predict.split(' ')
x_predict_test = []
x_predict_new = []
for word in x_predict:
        x_predict_new.append(word)
        x_predict_test.append(x_predict_new[:])

x_predict_1 = token.texts_to_sequences(x_predict_test)
x_predict_1 = pad_sequences(x_predict_1,padding='pre', maxlen=5)
predict = model.predict(x_predict_1)
score = 0
for i, word in enumerate(x_predict_test):
        print(word)
        print(predict[i])
        score = predict[i][0]
print(score)
if (score > 0.5):
        print("{:.2f}% 확률로 긍정 입니다.\n".format(score * 100))
else:
        print("{:.2f}% 확률로 부정 입니다.\n".format((1 - score) * 100))


"나는 반장이 정말 정말 재미 없다"
61.91% 확률로 부정 입니다.

"나는 반장이 정말 재미 없다 정말"
62.78% 확률로 부정 입니다.

"나는 반장이 정말 재미 있다 정말"
62.53% 확률로 부정 입니다.

"나는 재미 있다 "
'''
