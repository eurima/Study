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

le = LabelEncoder()
le.fit(train['type'])
x['type'] = le.transform(train['type'])


from tensorflow.keras.utils import to_categorical
# y = to_categorical(y) #<=============== class 개수대로 자동으로 분류 해 준다!!! /// 간단!!
#--to_categorical은 빈부분을 채우니 주의 [0. 0. 0. 0. 0. 0. 1. 0. 0.]
#-------------------------
y = np.array(y).reshape(-1,1)
enc= OneHotEncoder()   #[0. 0. 1. 0. 0.]
enc.fit(y)
y = enc.transform(y).toarray()

print(y.shape)


# print(np.unique(y))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.8, shuffle = True, random_state = 66) #

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = RobustScaler()
scaler = MaxAbsScaler()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)



#2 모델구성
#        
deep_len = [10, 8, 6, 4, 6, 4, 2, 30, 20, 10, 40, 30, 20, 10, 5, 2]
model = Sequential()
# model.add(Dense(deep_len[0],activation ='relu', input_dim = x_train.shape[1])) #activation = 'linear'
# model.add(Dense(deep_len[1]))
# model.add(Dense(deep_len[2]))
# model.add(Dense(deep_len[3],activation ='relu' )) 
# model.add(Dense(deep_len[4])) 
# model.add(Dense(deep_len[5],activation ='sigmoid'))
# model.add(Dense(deep_len[6],activation ='relu' )) 
# model.add(Dense(deep_len[7],activation ='sigmoid')) 
# model.add(Dense(deep_len[8],activation ='relu' )) 
# model.add(Dense(deep_len[9],activation ='sigmoid' )) 
# model.add(Dense(deep_len[10],activation ='relu'))
# model.add(Dense(deep_len[11])) 
# model.add(Dense(deep_len[12])) 
# model.add(Dense(deep_len[13],activation ='relu')) 
# model.add(Dense(deep_len[14])) 
# model.add(Dense(deep_len[15])) 

model.add(Dense(100,input_dim = x_train.shape[1])) #activation = 'linear'
model.add(Dense(50, activation ='relu' )) 
model.add(Dense(10, activation ='relu' )) 
model.add(Dense(5)) 
model.add(Dense(y.shape[1], activation = 'softmax'))
# sigmoid는 0 ~ 1 사이의 값을 뱉는다

#3. 컴파일, 훈련
epoch = 100
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy']) # metrics=['accuracy'] 영향을 미치지 않는다

from tensorflow.keras.callbacks import EarlyStopping
patience_num = 50
es = EarlyStopping(monitor='val_loss', patience=patience_num, mode = 'min', verbose=1, restore_best_weights=True)

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
test_flie = scaler.transform(test_flie)
# ############### 제출용.
result = model.predict(test_flie)
print(result[:5])

result_recover = np.argmax(result, axis = 1).reshape(-1,1) + 4
print(result_recover[:5])
print(np.unique(result_recover)) # np.unique()

submission['quality'] = result_recover

# # print(submission[:10])
submission.to_csv(path+"sampleHR.csv", index = False)
# print(result_recover)
acc= str(round(loss[1]*100,4))
model.save(f"./_save/keras24_dacon_save_model_{acc}.h5")

'''
Normal



'''