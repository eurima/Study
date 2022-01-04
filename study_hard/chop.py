import glob, pickle # 파일 불러오기에 유용한 라이브러리들
import numpy as np # 행렬 계산
# MIDI 파일을 다루기 위한 라이브러리
from music21 import converter, instrument, note, chord, stream
# 순차 모델 생성을 위한 라이브러리
from keras.models import Sequential
# LSTM : CPU 동작 / CuDNNLSTM : GPU 동작
from keras.layers import Dense, Dropout, LSTM, CuDNNLSTM
# One-hot Vector 만들기 위한 라이브러리
from keras.utils import np_utils

# midi = converter.parse("chpn-p9_format0.mid")
# # MIDI 파일 내의 notes(음정, 박자를 포함하는 정보)를 불러온다
# notes_to_parse = midi.flat.notes
# # 불러온 notes의 갯수
# print(np.shape(notes_to_parse))
# # 10개 테스트 출력
# for e in notes_to_parse[:20]:
#   print(e, e.offset)
# Note / Chord 두 종류로 나뉜다, Chord는 Note의 집합이다

# Chopin 폴더의 모든 MIDI 파일의 정보를 뽑아 하나로 만든다
# MIDI 파일로부터 Note 정보만 뽑아서 저장할 리스트
notes = []
# chopin 폴더 내의 모든 MIDI 파일에 반복문으로 접근
# glob.glob() : *를 제외한 나머지 부분이 같은 파일 이름들을 배열로 저장
# enumerate : 파일이름 배열을 순차적으로 하나씩 file에 넣는다
# i : 0 부터 1씩 증가 / file : 각 파일 이름
for i, file in enumerate(glob.glob("chopin/*.mid")):
  # midi: MIDI 파일의 전체 정보를 담고 있다 ------------------------------------------
  midi = converter.parse(file)
  print('\r', 'Parsing file ', i, " ", file, end='')  # 현재 진행 상황 출력
  # notes_to_parse : MIDI 파일을 Notes로 나누어 다루기 위한 변수
  notes_to_parse = None
  # try / except : try 수행 중 에러 발생 시 except 수행 -----------------------------
  # MIDI 파일 구조 차이로 인한 에러 방지
  # MIDI 파일의 Note / Chord / Tempo 정보만 가져온다
  try:  # file has instrument parts
    s2 = instrument.partitionByInstrument(midi)
    notes_to_parse = s2.parts[0].recurse()

  except:  # file has notes in a flat structure
    notes_to_parse = midi.flat.notes
  # Note / Chord / Tempo 정보 중 Note, Chord 의 경우 따로 처리, Tempo 정보는 무시 ----
  for e in notes_to_parse:
    # Note 인 경우 높이(Pitch), Octave 로 저장
    if isinstance(e, note.Note):
      notes.append(str(e.pitch))
    # Chord 인 경우 각 Note의 음높이(Pitch)를 '.'으로 나누어 저장
    elif isinstance(e, chord.Chord):
      # ':'.join([0, 1, 2]) : [0, 1, 2] -> [0:1:2]
      # str(n) for n in e.normalOrder
      #     => e.normalOrder 라는 배열 내의 모든 원소 n에 대해 str(n) 해준 새 배열을 만든다.
      #        ex) str(i) for i in [1, 2, 3] => ['1', '2', '3']
      notes.append('.'.join(str(n) for n in e.normalOrder))