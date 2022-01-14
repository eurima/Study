from sklearn.datasets import load_diabetes as load_data
import numpy as np
import time
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
#1 데이터
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
                                            ,('KNeighborsClassifier', model_4)
                                            ,('LogisticRegression', model_5)
                                            ,('DecisionTreeClassifier', model_6)
                                            ,('DecisionTreeClassifier', model_7)
                                            ], voting='hard')
classifiers = [model_1, model_2,model_3,model_4,model_5,model_6,model_7]
for classifier in classifiers:
    classifier.fit(x_train, y_train)
    class_name = classifier.__class__.__name__
    print("============== " + class_name + " ==================")
    
    result = classifier.score(x_test, y_test) 
    y_predict = classifier.predict(x_test)
    acc = accuracy_score(y_test, y_predict)
    print("result", result)
    print("accuracy_score", acc)


# voting_model.fit(x_train, y_train)
# pred = voting_model.predict(x_test)

end = time.time() - start
print('시간 : ', round(end,2) ,'초')
'''
============== Perceptron ==================
result 0.0
accuracy_score 0.0
============== LinearSVC ==================
result 0.0
accuracy_score 0.0
============== SVC ==================
result 0.0
accuracy_score 0.0
============== KNeighborsClassifier ==================
result 0.0
accuracy_score 0.0
============== LogisticRegression ==================
result 0.0
accuracy_score 0.0
============== DecisionTreeRegressor ==================
result -0.3002481344688075
accuracy_score 0.0
============== RandomForestClassifier ==================
result 0.011235955056179775
accuracy_score 0.011235955056179775
시간 :  0.77 초

Process finished with exit code 0


'''
