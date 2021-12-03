from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, LabelEncoder, OneHotEncoder

#1 데이터
path = "D:\\_data\\dacon\\wine\\" 
train = pd.read_csv(path +"train.csv")
test_flie = pd.read_csv(path + "test.csv") 

submission = pd.read_csv(path+"sample_Submission.csv") #제출할 값
y = train['quality']
x = train.drop(['quality'], axis =1) #
# x = train #.drop(['casual','registered','count'], axis =1) #

le = LabelEncoder()
le.fit(train['type'])
# x_type = le.transform(train['type'])
# x = x.drop(['type'], axis = 1)
# x = pd.concat([x,x_type])
x['type'] = le.transform(train['type'])
# print(x)
# y = np.array(y).reshape(-1,1)
# one_hot = OneHotEncoder()
# one_hot.fit(y)
# y = one_hot.transform(y).toarray()
# print(y)



from tensorflow.keras.utils import to_categorical
# one_hot = to_categorical(y,num_classes=len(np.unique(y)))
y = to_categorical(y) #<=============== class 개수대로 자동으로 분류 해 준다!!! /// 간단!!
print(y)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.8, shuffle = True, random_state = 66) #

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = MaxAbsScaler()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)



#2 모델구성
#        
deep_len = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 40, 30, 20, 10, 5, 2]
model = Sequential()
model.add(Dense(deep_len[0], activation = 'linear', input_dim =x.shape[1]))
model.add(Dense(deep_len[1], )) # ===> 디폴트 값은 linear이고 sigmoid를 넣을 수도 있다 (값이 튀다면 sigmoid로 한번씩 잡아주면 성능이 좋아질 수 있다)
model.add(Dense(deep_len[2]))
model.add(Dense(deep_len[3],activation ='relu')) 
model.add(Dense(deep_len[4])) 
model.add(Dense(deep_len[5],activation ='relu'))
model.add(Dense(deep_len[6])) 
model.add(Dense(deep_len[7],activation ='relu')) 
model.add(Dense(deep_len[8])) 
model.add(Dense(deep_len[9])) 
# model.add(Dense(deep_len[10],activation ='relu'))
# model.add(Dense(deep_len[11])) 
# model.add(Dense(deep_len[12])) 
# model.add(Dense(deep_len[13],activation ='relu')) 
# model.add(Dense(deep_len[14])) 
# model.add(Dense(deep_len[15])) 
model.add(Dense(y.shape[1], activation = 'softmax'))
# sigmoid는 0 ~ 1 사이의 값을 뱉는다

#3. 컴파일, 훈련
epoch = 10000
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy']) # metrics=['accuracy'] 영향을 미치지 않는다

from tensorflow.keras.callbacks import EarlyStopping
patience_num = 50
es = EarlyStopping(monitor='val_loss', patience=patience_num, mode = 'auto', verbose=1, restore_best_weights=True)

start = time.time()

model.fit(x_train, y_train, epochs = epoch, batch_size =1,validation_split=0.2,callbacks=[es])
end = time.time() - start
print('시간 : ', round(end,2) ,'초')

#4 평가예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss[0]) #<==== List 형태로 제공된다
print("accuracy : ",loss[1])
print("epochs :",epoch)


test_flie['type'] = le.transform(test_flie['type'])


scaler.transform(test_flie)
# ############### 제출용.
result = model.predict(test_flie)
result_recover = np.argmax(y, axis =1).reshape(-1,1)
submission['quality'] = result_recover

# # print(submission[:10])
submission.to_csv(path+"sampleHR_MinMaxScaler.csv", index = False)
print(result_recover)

'''
Normal


MinMaxScaler



StandardScaler



RobustScaler



MaxAbsScaler






'''