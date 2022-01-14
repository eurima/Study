from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris, load_wine, load_diabetes, load_boston, load_breast_cancer, fetch_covtype
import warnings
import numpy as np

from sympy import N
warnings.filterwarnings(action='ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 머신러닝 모델 구성 
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsClassifier
from sklearn.linear_model import LogisticRegression #이름은 그렇지만 분류! 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

dataset  = load_iris()
x = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
n_split = 5
# kfold = KFold(n_splits=n_split, shuffle=True, random_state=66)
kfold = StratifiedKFold(n_splits=n_split,shuffle=True, random_state=66)
model = SVC()

scores = cross_val_score(model,x,y,cv=kfold)
print("ACC :", scores,"\n", round(np.mean(scores),4)) #ACC : [0.96666667 0.96666667 1.         0.93333333 0.96666667]

