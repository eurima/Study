from sklearn.datasets import load_iris
import numpy as np
import time
#1 데이터
dataset  = load_iris()
'''
    :Number of Instances: 150 (50 in each of three classes)
    :Number of Attributes: 4 numeric, predictive attributes and the class
    :Attribute Information:
        - sepal length in cm
        - sepal width in cm
        - petal length in cm
        - petal width in cm
        - class:
                - Iris-Setosa
                - Iris-Versicolour
                - Iris-Virginica
                
x=(150,4), y= (150,1)
'''
x = dataset.data
y = dataset.target
# print(x.shape, y.shape) #(150,4) (150,)
# print(np.unique(y)) #----> [0, 1,2] : 배열의 고유값을 찾아준다 (라벨값이 어떤것이 있는가) len(np.unique(y))
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y) --------> 머신러닝은 원핫 인코딩 불필요
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.8, shuffle = True, random_state = 66) #455.2 /114
# 머신러닝 모델구성 
from sklearn.svm import LinearSVC
model = LinearSVC()
start = time.time()
# 머신러닝 훈련
model.fit(x_train, y_train)
# 머신러닝 평가 예측
result = model.score(x_test, y_test) #분류 acc, 회귀 R2 자동으로 출력

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print("result", result)
print("accuracy_score", acc)

end = time.time() - start
print('시간 : ', round(end,2) ,'초')

'''
#2 딥러닝모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
      
deep_len = [100, 50, 30, 20, 100, 50, 30, 40, 50, 40, 30, 20, 10, 5, 4, 2]
model = Sequential()
model.add(Dense(deep_len[0], activation = 'linear', input_dim =x.shape[1]))
model.add(Dense(deep_len[1], )) # ===> 디폴트 값은 linear이고 sigmoid를 넣을 수도 있다 (값이 튀다면 sigmoid로 한번씩 잡아주면 성능이 좋아질 수 있다)
model.add(Dense(deep_len[2]))
model.add(Dense(deep_len[3])) 
model.add(Dense(deep_len[4])) 
model.add(Dense(deep_len[5])) 
model.add(Dense(deep_len[6])) 
model.add(Dense(deep_len[7])) 
model.add(Dense(deep_len[8])) 
model.add(Dense(deep_len[9])) 
model.add(Dense(deep_len[10])) 
model.add(Dense(deep_len[11])) 
model.add(Dense(deep_len[12])) 
model.add(Dense(deep_len[13])) 
model.add(Dense(deep_len[14])) 
model.add(Dense(deep_len[15])) 
model.add(Dense(y.shape[1], activation = 'softmax')) #이진분류의 마지막 레이어는 무조건 sigmoid!!!!
# sigmoid는 0 ~ 1 사이의 값을 뱉는다


#3. 컴파일, 훈련
epoch = 10000
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy']) # metrics=['accuracy'] 영향을 미치지 않는다

from tensorflow.keras.callbacks import EarlyStopping
patience_num = 50
es = EarlyStopping(monitor='val_loss', patience=patience_num, mode = 'auto', verbose=1, restore_best_weights=True)
# 통상 val_loss 가 성능이 더 좋다
# 그렇지만 너무 튄다면 loss를 넣어도 된다
# 나중에는 monitor='accuracy' ,mode 가 헷갈리면 'auto'로 잡는다
start = time.time()

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = MaxAbsScaler()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)

model.fit(x_train, y_train, epochs = epoch, batch_size =1,validation_split=0.2,callbacks=[es])
end = time.time() - start
print('시간 : ', round(end,2) ,'초')

#4 평가예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss[0]) #<==== List 형태로 제공된다
print("accuracy : ",loss[1])
#===========> 가장 중요한것은 Loss 이다!!
#===========> Loss 가 가장 낮은 모델이 무조건 좋은 것이다!!!
#===========> 더욱 중요한것은 val_loss 이다!!
y_predict = model.predict(x_test)
print("epochs :",epoch)
result = y_predict[:7]
print(result)
print(y_test[:7])
'''
'''
결과>
딥러닝
Epoch 00196: early stopping
시간 :  162.39 초
loss :  0.06246219575405121
accuracy :  0.9666666388511658
------------------------------
SVC>
result 0.9666666666666667
accuracy_score 0.9666666666666667
시간 :  0.04 초
'''
