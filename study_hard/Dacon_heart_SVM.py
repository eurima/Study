import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import metrics
from sklearn.svm import SVR
import warnings
warnings.filterwarnings(action='ignore')

# import mglearn
# %matplotlib inline

def f1_score(answer, submission):
    true = answer
    pred = submission
    score = metrics.f1_score(y_true=true, y_pred=pred)
    return str(score)

def to_binary_list(pred_val_list):
    val_list = []
    for i in pred_val_list:
        if i < 0.5:
            val_list.append(0)
        else:
            val_list.append(1)
    return val_list

#1 데이터
path = "D:\\Study\_data\heart\\" #D:\Study\_data\heart
train = pd.read_csv(path +"train.csv")
test_file = pd.read_csv(path + "test.csv")
submission = pd.read_csv(path+"sample_Submission.csv")

y = train['target']
x = train.drop(['id','target'], axis =1)
test_file = test_file.drop(['id'],axis =1)

x = x.drop(['restecg','chol','fbs'],axis =1)
test_file = test_file.drop(['restecg','chol','fbs'],axis =1)

x['ca'] = x['ca'].replace(4,0)
test_file['ca'] = test_file['ca'].replace(4,0)

x['thal'] = x['thal'].replace(0,2)
test_file['thal'] = test_file['thal'].replace(0,2)

from sklearn import decomposition
dreduction = decomposition.PCA(n_components=1)
dreduction.fit(x)
x = dreduction.transform(x)
test_file = dreduction.transform(test_file)
#print(x.shape) #151,1

####################

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.33)
x_train, x_validation, y_train, y_validation = model_selection.train_test_split(x_train, y_train, test_size=0.33)

estimator_svr_linear = SVR(kernel='linear') #linear SVM
estimator_svr_linear.fit(x_train, y_train)

estimator_svr_poly = SVR(kernel='poly') #polynomial SVM
estimator_svr_poly.fit(x_train, y_train)

estimator_svr_rbf = SVR() #rbf SVM #디폴트
estimator_svr_rbf.fit(x_train, y_train)

import matplotlib.pylab as plt
import matplotlib as mpl
mpl.rc('font', family='Malgun Gothic') #한글 폰트 설정
plt.scatter(x[:,0], y, c='red')
#
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
x_plot = np.linspace(xmin,xmax)
#'''
y_perdict_svr_linear = estimator_svr_linear.predict(x_plot.reshape((-1,1)))
y_perdict_svr_poly = estimator_svr_poly.predict(x_plot.reshape((-1,1)))
y_perdict_svr_rbf = estimator_svr_rbf.predict(x_plot.reshape((-1,1)))
plt.plot(x_plot, y_perdict_svr_linear, color='red', label='linear SVM')
plt.plot(x_plot, y_perdict_svr_poly, color='green', label='polynomial SVM')
plt.plot(x_plot, y_perdict_svr_rbf, color='blue', label='rbf SVM')
#'''
#
plt.title('선 그래프')
plt.xlabel('x')
plt.ylabel('y')
plt.legend() #범례
plt.show()

estimator_svr_linear_results = to_binary_list(estimator_svr_linear.predict(test_file))
estimator_svr_linear_num = f1_score(y_test,to_binary_list(estimator_svr_linear.predict(x_test)))
estimator_svr_poly_results = to_binary_list(estimator_svr_poly.predict(test_file))
estimator_svr_poly_num = f1_score(y_test,to_binary_list(estimator_svr_poly.predict(x_test)))
estimator_svr_rbf_results = to_binary_list(estimator_svr_rbf.predict(test_file))
estimator_svr_rbf_num = f1_score(y_test,to_binary_list(estimator_svr_rbf.predict(x_test)))
print(estimator_svr_linear_num,estimator_svr_poly_num,estimator_svr_rbf_num)
print("estimator_svr_linear_results",estimator_svr_linear_results)
print("estimator_svr_poly_results",estimator_svr_poly_results)
print("estimator_svr_rbf_results",estimator_svr_rbf_results)


