# import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# X, y = zip(['a', 1], ['b', 2], ['c', 3])
# print('X 데이터 :',X)
# print('y 데이터 :',y)
# sequences = [['a', 1], ['b', 2], ['c', 3]]
# X, y = zip(*sequences)
# print('X 데이터 :',X)
# print('y 데이터 :',y)
#
# values = [['당신에게 드리는 마지막 혜택!', 1],
# ['내일 뵐 수 있을지 확인 부탁드...', 0],
# ['도연씨. 잘 지내시죠? 오랜만입...', 0],
# ['(광고) AI로 주가를 예측할 수 있다!', 1]]
# columns = ['메일 본문', '스팸 메일 유무']
#
# df = pd.DataFrame(values, columns=columns)
# print(df)
# X = df['메일 본문']
# y = df['스팸 메일 유무']
#
#
# print('X 데이터 :',X.to_list())
# print('y 데이터 :',y.to_list())
# np_array = np.arange(0,16).reshape((4,4))
# # print('전체 데이터 :')
# # print(np_array)
# X = np_array[:, :3]
# y = np_array[:,3]
# #마지막 열을 제외하고 X데이터에 저장합니다. 마지막 열만을 y데이터에 저장합니다.
# print('X 데이터 :')
# print(X)
# print('y 데이터 :',y)
#사이킷런은 학습용 테스트와 테스트용 데이터를 쉽게 분리할 수 있게 해주는 train_test_split()를 지원합니다.
# 임의로 X와 y 데이터를 생성
X, y = np.arange(10).reshape((5, 2)), range(5)

print('X 전체 데이터 :')
print(X)
print('y 전체 데이터 :')
print(list(y))
# 7:3의 비율로 훈련 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
print('X 훈련 데이터 :')
print(X_train)
print('X 테스트 데이터 :')
print(X_test)