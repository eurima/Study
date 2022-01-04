import numpy as np
from music21 import * #pip install music21
#xml 악보가 아닌 오선지 악보를 이용하기 위하여 MusicScore 패키지를 다운로드 받아서 설치한다.
# https://musescore.org 에서 FreeDownload
# MuseScore-3.5.2.3.. msi 실행
# path 설정 한번 만 샐행
# us=environment.UserSettings()
# us["musescoreDirectPNGPath"]=r"C:\Program Files\MuseScore 3\bin\MuseScore3.exe"
# us["musicxmlPath"]=r"C:\Program Files\MuseScore 3\bin\MuseScore3.exe"
s=converter.parse("tinynotation: 3/4 c4 d8 f g16 a g f#") #<music21.stream.Part0x21cacfd32c8>
# s.show('midi', fp='sample.mid') #음악재생 및 sample.mid로 저장
# s.show() #오선악보 출력
p=converter.parse("tinynotation: 3/4 c4 d8 f g16 a g f#")
for n in list(p.flat.notes) : #음표출력
    print("Note: {}{} {:0.1f}".format(n.pitch.name, n.pitch.octave, n.duration.quarterLength))


