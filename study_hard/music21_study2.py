# from music21 import *
# c=converter.parse("""tinynotation: 4/4
# g4 g e8 f8 g4 a a g2 g4 c' e' d'8 c'8 d'.2 r4
# e'4 e'4 d' d' c' d'8 c'8 a4 a4 g4 g g e8 d8 c.2 r4
# d4 d e c d4 d e g g4 c' e' d'8 c'8 d'.2 r4
# e'4 e' d' d' c' d'8 c'8 a4 a4 g4 g g e8 d8 c.2 r4"
# """)
# c.show('midi',fp='고향의봄.mid')
# c.show()
import numpy
from music21 import converter, instrument, note, chord
import glob    # 원문에는 없지만 아래에서 사용하기 때문에 glob을 import 해줘야합니다.

notes = []
for file in glob.glob("midi_songs/*.mid"):
    midi = converter.parse(file)
    notes_to_parse = None
    try:       # 학습 데이터 중 TypeError 를 일으키는 파일이 있어서 해놓은 예외처리
        parts = instrument.partitionByInstrument(midi)
    except TypeError:
        print('## 1 {} file occur error.'.format(file))
    if parts: # file has instrument parts
        print('## 2 {} file has instrument parts'.format(file))
        notes_to_parse = parts.parts[0].recurse()
    else: # file has notes in a flat structure
        print('## 3 {} file has notes in a flat structure'.format(file))
        notes_to_parse = midi.flat.notes
    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))
sequence_length = 100
# 모든 계이름의 이름을 pitchnames 변수에 저장.
# set 으로 중복을 피하고, sorted 함수로 정렬함.
pitchnames = sorted(set(item for item in notes))

# 각 계이름을 숫자로 바꾸는 dictionary(사전)을 만든다.
note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
network_input = []
network_output = []

# 입력 시퀀스를 만든다.
for i in range(0, len(notes) - sequence_length, 1):
    sequence_in = notes[i:i + sequence_length]
    sequence_out = notes[i + sequence_length]
    network_input.append([note_to_int[char] for char in sequence_in])
    network_output.append(note_to_int[sequence_out])

n_patterns = len(network_input)

# 데이터 입력 형태를 LSTM 레이어에 알맞게 변경함.
network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))

# 입력값을 normalizing(정규화)
network_input = network_input / float(n_vocab)
network_output = np_utils.to_categorical(network_output)