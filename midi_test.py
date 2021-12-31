import glob, pickle # 파일 불러오기에 유용한 라이브러리들
import numpy as np #
# MIDI 파일을 다루기 위한 라이브러리
from music21 import converter, instrument, note, chord, stream
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, CuDNNLSTM

midi_path = "D:\\midi_midi\\"
midi_file = "BEATLES_THE_-_A_day_in_the_life.mid"

midi = converter.parse(midi_path+midi_file)

# MIDI 파일 내의 notes(음정, 박자를 포함하는 정보)를 불러온다
notes_to_parse = midi.flat.notes
# 불러온 notes의 갯수
print(np.shape(notes_to_parse))

for e in notes_to_parse[:20]:
  print(e, e.offset)

