import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import reuters

(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=None, test_split=0.2)

# print('훈련용 뉴스 기사 : {}'.format(len(X_train)))
# print('테스트용 뉴스 기사 : {}'.format(len(X_test)))
# num_classes = len(set(y_train))
# print('카테고리 : {}'.format(num_classes))
'''
훈련용 뉴스 기사 : 8982
테스트용 뉴스 기사 : 2246
카테고리 : 46
'''
word_to_index = reuters.get_word_index()
# print(word_to_index)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

vocab_size = 1000
max_len = 100

X_train = pad_sequences(x_train, maxlen=max_len)
X_test = pad_sequences(x_test, maxlen=max_len)
#
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

word_to_index = reuters.get_word_index()
import operator
# print(sorted(word_to_index.items(),key = operator.itemgetter(1)))

# index_to_word ={}
# for key, value in word_to_index.items():
#     index_to_word[value+3] = key

# for index, token in enumerate(("<pad>", "<sos>", "<unk>")):
#       index_to_word[index] = token

# print(' '.join([index_to_word[index] for index in x_train[0]]))
