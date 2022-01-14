from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris, load_wine, load_diabetes, load_boston, load_breast_cancer
import warnings
warnings.filterwarnings(action='ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# datasets = {'Iris':load_iris(),'Wine':load_wine(),'Diabets':load_diabetes(),'Cancer':load_breast_cancer(),'Boston':load_boston()}
dataset = load_wine()
alAlgorithms_classifier = all_estimators(type_filter='classifier')#classifier'41 regressor 55, transformer 79, cluster 10
alAlgorithms_regressor = all_estimators(type_filter='regressor')

num = 0

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y,
     train_size = 0.8, shuffle = True, random_state = 66)

scaler = MinMaxScaler()
scaler.fit_transform(x_train)
scaler.transform(x_test)
print('--alAlgorithms_classifier')
for (name, algo) in alAlgorithms_classifier:
    try:
        model = algo()
        model.fit(x_train,y_train)
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test, y_predict)
        print(name, '의 정답률 : ',acc)

    except:
        print(name,' Err--------------------------')
        num = num + 1
print('---------------------------------')
# print('모델의 갯수 : ',len(alAlgorithms_classifier))
print('classifier 안되는 모델의 갯수 : ',num)

for (name, algo) in alAlgorithms_regressor:
    try:
        model = algo()
        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test, y_predict)
        print(name, '의 정답률 : ', acc)

    except:
        print(name, ' Err')
        num = num + 1
print('---------------------------------')
# print('모델의 갯수 : ',len(alAlgorithms_classifier))
print('regressor 안되는 모델의 갯수 : ', num)

'''
--alAlgorithms_classifier
AdaBoostClassifier 의 정답률 :  0.8888888888888888
BaggingClassifier 의 정답률 :  1.0
BernoulliNB 의 정답률 :  0.4166666666666667
CalibratedClassifierCV 의 정답률 :  0.9444444444444444
CategoricalNB  Err--------------------------
ClassifierChain  Err--------------------------
ComplementNB 의 정답률 :  0.6944444444444444
DecisionTreeClassifier 의 정답률 :  0.9722222222222222
DummyClassifier 의 정답률 :  0.4166666666666667
ExtraTreeClassifier 의 정답률 :  0.9444444444444444
ExtraTreesClassifier 의 정답률 :  1.0
GaussianNB 의 정답률 :  1.0
GaussianProcessClassifier 의 정답률 :  0.4444444444444444
GradientBoostingClassifier 의 정답률 :  0.9722222222222222
HistGradientBoostingClassifier 의 정답률 :  0.9722222222222222
KNeighborsClassifier 의 정답률 :  0.6944444444444444
LabelPropagation 의 정답률 :  0.5277777777777778
LabelSpreading 의 정답률 :  0.5277777777777778
LinearDiscriminantAnalysis 의 정답률 :  1.0
LinearSVC 의 정답률 :  0.8888888888888888
LogisticRegression 의 정답률 :  0.9722222222222222
LogisticRegressionCV 의 정답률 :  1.0
MLPClassifier 의 정답률 :  0.8055555555555556
MultiOutputClassifier  Err--------------------------
MultinomialNB 의 정답률 :  0.7777777777777778
NearestCentroid 의 정답률 :  0.6944444444444444
NuSVC 의 정답률 :  0.9444444444444444
OneVsOneClassifier  Err--------------------------
OneVsRestClassifier  Err--------------------------
OutputCodeClassifier  Err--------------------------
PassiveAggressiveClassifier 의 정답률 :  0.5555555555555556
Perceptron 의 정답률 :  0.6388888888888888
QuadraticDiscriminantAnalysis 의 정답률 :  0.9722222222222222
RadiusNeighborsClassifier  Err--------------------------
RandomForestClassifier 의 정답률 :  1.0
RidgeClassifier 의 정답률 :  1.0
RidgeClassifierCV 의 정답률 :  1.0
SGDClassifier 의 정답률 :  0.6944444444444444
SVC 의 정답률 :  0.6944444444444444
StackingClassifier  Err--------------------------
VotingClassifier  Err--------------------------
---------------------------------
classifier 안되는 모델의 갯수 :  9
DecisionTreeRegressor 의 정답률 :  0.9444444444444444
ExtraTreeRegressor 의 정답률 :  0.8333333333333334
RadiusNeighborsRegressor 의 정답률 :  0.0
---------------------------------
regressor 안되는 모델의 갯수 :  61

Process finished with exit code 0


'''