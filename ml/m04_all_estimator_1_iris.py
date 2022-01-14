from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris, load_wine, load_diabetes, load_boston, load_breast_cancer
import warnings
warnings.filterwarnings(action='ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# datasets = {'Iris':load_iris(),'Wine':load_wine(),'Diabets':load_diabetes(),'Cancer':load_breast_cancer(),'Boston':load_boston()}
dataset = load_iris()
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
        #print(name,' Err')
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
        #print(name, ' Err')
        num = num + 1
print('---------------------------------')
# print('모델의 갯수 : ',len(alAlgorithms_classifier))
print('regressor 안되는 모델의 갯수 : ', num)

'''
--alAlgorithms_classifier
AdaBoostClassifier 의 정답률 :  0.6333333333333333
BaggingClassifier 의 정답률 :  0.9666666666666667
BernoulliNB 의 정답률 :  0.3
CalibratedClassifierCV 의 정답률 :  0.9
CategoricalNB 의 정답률 :  0.9
ComplementNB 의 정답률 :  0.6666666666666666
DecisionTreeClassifier 의 정답률 :  0.9333333333333333
DummyClassifier 의 정답률 :  0.3
ExtraTreeClassifier 의 정답률 :  0.9333333333333333
ExtraTreesClassifier 의 정답률 :  0.9666666666666667
GaussianNB 의 정답률 :  0.9666666666666667
GaussianProcessClassifier 의 정답률 :  0.9666666666666667
GradientBoostingClassifier 의 정답률 :  0.9666666666666667
HistGradientBoostingClassifier 의 정답률 :  0.8666666666666667
KNeighborsClassifier 의 정답률 :  0.9666666666666667
LabelPropagation 의 정답률 :  0.9333333333333333
LabelSpreading 의 정답률 :  0.9333333333333333
LinearDiscriminantAnalysis 의 정답률 :  1.0
LinearSVC 의 정답률 :  0.9666666666666667
LogisticRegression 의 정답률 :  1.0
LogisticRegressionCV 의 정답률 :  1.0
MLPClassifier 의 정답률 :  1.0
MultinomialNB 의 정답률 :  0.9666666666666667
NearestCentroid 의 정답률 :  0.9333333333333333
NuSVC 의 정답률 :  0.9666666666666667
PassiveAggressiveClassifier 의 정답률 :  0.7
Perceptron 의 정답률 :  0.9333333333333333
QuadraticDiscriminantAnalysis 의 정답률 :  1.0
RadiusNeighborsClassifier 의 정답률 :  0.9666666666666667
RandomForestClassifier 의 정답률 :  0.9333333333333333
RidgeClassifier 의 정답률 :  0.8666666666666667
RidgeClassifierCV 의 정답률 :  0.8666666666666667
SGDClassifier 의 정답률 :  0.6666666666666666
SVC 의 정답률 :  0.9666666666666667
---------------------------------
classifier 안되는 모델의 갯수 :  7
DecisionTreeRegressor 의 정답률 :  0.9333333333333333
ExtraTreeRegressor 의 정답률 :  0.9333333333333333
---------------------------------
regressor 안되는 모델의 갯수 :  60

Process finished with exit code 0


'''