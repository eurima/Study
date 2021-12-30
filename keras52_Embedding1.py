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
# print(token.word_index)
#{'참': 1, '너무': 2, '잘': 3, '재밋어요': 4, '최고예요': 5,
# '만든': 6, '영화예요': 7, '추천하고': 8, '싶은': 9, '영화입니다': 10,
# '한': 11, '번': 12, '더': 13, '보고': 14, '싶네요': 15, '글쎄요': 16,
# '별로예요': 17, '생각보다': 18, '지루해요': 19, '연기가': 20, '어색해요': 21,
# '재미없어요': 22, '재미없다': 23, '재밋네요': 24, '예람이가': 25, '생기긴': 26, '했어요': 27}
x= token.texts_to_sequences(docs)
# print(x)
#[[2, 4], [1, 5], [1, 3, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17], [18, 19], [20, 21], [22], [2, 23], [1, 24], [25, 3, 26, 27]]

from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre', maxlen=5) #-> input shape를 맞추기 위해
# print(pad_x.shape) #(13,5)

word_size = len(token.word_index)
# print("word_size : ",word_size)#27
# print(len(np.unique(pad_x))) # 28 -> 0이 붙었기 때문
#----> 원핫 인코딩을 하면 (13,5) -> (13,5,28) 이 되버린다
# 옥스포드 사전은? 13,5,1000000 -> 6500만개?!
# 그래서 원핫 인코딩 하지말고 Embedding 해야 한다
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM

#2 모델                                            인풋은 (13,5)
model = Sequential()#단어 사전의 개수               단어수, 길이
# model.add(Embedding(input_dim=28, output_dim=10, input_length=5))#Embedding 레이어는 연산이 아니라 벡터화해주는 역할
model.add(Embedding(100, 10))#단어사전의 개수만 알면 돌아간다 2.7버전은 에러 없이 처리됨
# input_dim * output_dim 개의 벡터로 변환
model.add(LSTM(32))
model.add(Dense(1,activation='sigmoid'))
model.summary()
'''
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       (None, 5, 10)             280       
 lstm (LSTM)                 (None, 32)                5504      
 dense (Dense)               (None, 1)                 33         
=================================================================
Total params: 5,817
Trainable params: 5,817
Non-trainable params: 0
_________________________________________________________________

Process finished with exit code 0

'''

#3. 컴파일, 훈련
import time
opt="adam"
model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics=['acc']) # metrics=['accuracy'] 영향을 미치지 않는다

hist = model.fit(pad_x, lable, epochs = 100, batch_size=32)

########################################################################
#4 평가예측
from sklearn.metrics import r2_score, accuracy_score

acc = model.evaluate(pad_x,lable)[1]
print("acc : ",acc)
'''
acc :  1.0
'''
