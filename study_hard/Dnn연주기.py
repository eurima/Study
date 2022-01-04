import numpy as np
from music21 import *
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.utils import np_utils
from music21 import *
import json
TIME_STEP=5
고향봄 = """g4 g e8 f8 g4 a a g2 g4 c' e' d'8 c'8 d'.2 r4
e'4 e'4 d' d' c' d'8 c'8 a4 a4 g4 g g e8 d8 c.2 r4
d4 d e c d4 d e g g4 c' e' d'8 c'8 d'.2 r4
e'4 e' d' d' c' d'8 c'8 a4 a4 g4 g g e8 d8 c.2 r4"
"""
code2idx = {'c4':0, 'd4':1, 'e4':2, 'f4':3, 'g4':4, 'a4':5, 'b4':6,
            'c8':7, 'd8':8, 'e8':9, 'f8':10, 'g8':11, 'a8':12, 'b8':13}
def seq2dataset(seq, windowSize):
    dataset =[]
    for i in range(len(seq)-windowSize):
        subset = seq[i:(i+windowSize+1)]
        dataset.append([code2idx[item] for item in subset] )
    return np.array(dataset)

# 악보로부터 학습용 데이터 X와 라벨 Y 작성
dataset,n2i=seq2dataset(고향봄)
i2n={n2i[n]:n for n in n2i.keys()}