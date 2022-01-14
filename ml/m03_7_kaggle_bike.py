from sklearn.datasets import load_breast_cancer as load_data
import numpy as np
import pandas as pd

import time
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score, mean_squared_error

#1 데이터
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))


path = "D:\\Study\\_data\\bike\\"
train = pd.read_csv(path + "train.csv")

y = train['count']
x = train.drop(['casual','registered','count'], axis =1) #

x['datetime'] = pd.to_datetime(x['datetime'])
x['year'] = x['datetime'].dt.year
x['month'] = x['datetime'].dt.month
x['day'] = x['datetime'].dt.day
x['hour'] = x['datetime'].dt.hour
x = x.drop('datetime', axis=1)
#
#
# test_flie['datetime'] = pd.to_datetime(test_flie['datetime'])
# test_flie['year'] = test_flie['datetime'].dt.year
# test_flie['month'] = test_flie['datetime'].dt.month
# test_flie['day'] = test_flie['datetime'].dt.day
# test_flie['hour'] = test_flie['datetime'].dt.hour
# test_flie = test_flie.drop('datetime', axis=1)


# 로그변환
y = np.log1p(y) #----------> 로그변환 하면서 +1 해줌

dataset  = load_data()

x = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.8, shuffle = True, random_state = 66)
# 머신러닝 모델 구성 
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression #이름은 그렇지만 분류! 
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier

model_1 = Perceptron()
model_2 = LinearSVC()
model_3 = SVC()
model_4 = KNeighborsClassifier()
model_5 = LogisticRegression()
model_6 = DecisionTreeClassifier()
model_7 = RandomForestClassifier()

start = time.time()
# 머신러닝 훈련
voting_model = VotingClassifier(estimators=[ ('Perceptron', model_1)
                                            , ('LinearSVC', model_2)
                                            ,('SVC', model_3)
                                            ,('KNeighborsRegressor', model_4)
                                            ,('LogisticRegression', model_5)
                                            ,('DecisionTreeRegressor', model_6)
                                            ,('RandomForestClassifier', model_7)
                                            ], voting='hard')

classifiers = [model_1, model_2,model_3,model_4,model_5,model_6,model_7]

for classifier in classifiers:
    classifier.fit(x_train, y_train)
    class_name = classifier.__class__.__name__
    print("============== " + class_name + " ==================")
    
    result = classifier.score(x_test, y_test) 
    y_predict = classifier.predict(x_test)
    acc = accuracy_score(y_test, y_predict)

    r2 = r2_score(y_test, y_predict)
    rmse = RMSE(y_test, y_predict)
    print("R2 : ", r2)
    print("RMSE : ", rmse)

    print("result", result)
    print("accuracy_score", acc)


# voting_model.fit(x_train, y_train)
# pred = voting_model.predict(x_test)

end = time.time() - start
print('시간 : ', round(end,2) ,'초')
'''
============== Perceptron ==================
R2 :  0.5429335115268958
RMSE :  0.3244428422615251
result 0.8947368421052632
accuracy_score 0.8947368421052632
============== LinearSVC ==================
R2 :  0.35248914132976916
RMSE :  0.38616422286061647
result 0.8508771929824561
accuracy_score 0.8508771929824561
============== SVC ==================
R2 :  0.5429335115268958
RMSE :  0.3244428422615251
result 0.8947368421052632
accuracy_score 0.8947368421052632
============== KNeighborsClassifier ==================
R2 :  0.6572001336451719
RMSE :  0.28097574347450816
result 0.9210526315789473
accuracy_score 0.9210526315789473
============== LogisticRegression ==================
R2 :  0.8095556298028733
RMSE :  0.20942695414584775
result 0.956140350877193
accuracy_score 0.956140350877193
============== DecisionTreeClassifier ==================
R2 :  0.7714667557634479
RMSE :  0.22941573387056177
result 0.9473684210526315
accuracy_score 0.9473684210526315
============== RandomForestClassifier ==================
R2 :  0.8095556298028733
RMSE :  0.20942695414584775
result 0.956140350877193
accuracy_score 0.956140350877193
시간 :  0.23 초
'''
