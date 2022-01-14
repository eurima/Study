from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris, load_wine, load_diabetes, load_boston, load_breast_cancer, fetch_covtype
import warnings
warnings.filterwarnings(action='ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# datasets = {'Iris':load_iris(),'Wine':load_wine(),'Diabets':load_diabetes(),'Cancer':load_breast_cancer(),'Boston':load_boston()}
dataset = fetch_covtype()
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


'''