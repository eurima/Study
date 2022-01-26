import pandas as pd
import time
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import fetch_california_housing, load_boston, load_diabetes
from sklearn.model_selection import learning_curve, train_test_split, GridSearchCV

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, accuracy_score
import warnings
# warnings.filterwarnings('ignore')

#1. 데이터

datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=66)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
#훈련
histt = model = XGBRegressor(
    n_estimators=10000,       
    learning_rate = 0.01,    
    gamma = 1,       
    max_depth = 20,    
    min_child_weight = 5,    
    reg_lambda = 1,    
    reg_alpha = 0, 
    n_jobs=-1 
)
start = time.time()
model.fit(x_train, y_train, verbose=1,
          eval_set = [(x_train, y_train),(x_test,y_test)],
          eval_metric = 'rmse',
          early_stopping_rounds= 10
          )
end = time.time()-start
print("걸린 시간 : ", round(end, 2))
score = round(model.score(x_test, y_test),5)
print("score : ", score)

# y_pred = model.predict(x_test)
# r2 = round(r2_score(y_test, y_pred),2)
# print("r2 : ", r2)
results = model.evals_result()

# import matplotlib.pyplot as plt

# train_error = results['validation_0']['rmse']
# test_error = results['validation_1']['rmse']

# # epoch = range(1, len(train_error)+1)
# plt.plot(train_error, label = 'Train')
# plt.plot(test_error, label = 'Test')
# plt.ylabel('Classification Error')
# plt.xlabel('Model Complexity (n_estimators)')
# plt.legend()
# plt.show()
print(histt.get_params)