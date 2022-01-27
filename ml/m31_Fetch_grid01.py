import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
import warnings
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

import os
warnings.filterwarnings(action='ignore')    
report_path = 'D:\Study\_Report\\'#'D\\Study\_Report\\
file_name = os.path.abspath( __file__ ).split('\\')[3].split('.')[0]
file = open(report_path + file_name +".txt", "w")
x_train = np.load('fetch_covtype_datax_x_train.npy')
y_train = np.load('fetch_covtype_datax_y_train.npy')
x_test = np.load('fetch_covtype_datax_x_test.npy')
y_test = np.load('fetch_covtype_datax_y_test.npy')

param_grid={
            'booster' :['gbtree'],
            'tree_method' : ['gpu_hist'],
            'predictor':['gpu_predictor'],
            'gpu_id':[0],           
                #  'silent':[True],
                 'max_depth':[25],#range (10, 20, 5),
                 'min_child_weight':[5],
                 'gamma':[1],
                #  'nthread':[4],
                 'colsample_bytree':[0.5],
                 'colsample_bylevel':[0.9],
                 'n_estimators':[100000],
                 'objective':['mlogloss'],
                 'learning_rate':np.arange(0.005),
                 'random_state':[2]}

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2 모델
model = GridSearchCV(XGBClassifier(), param_grid, cv=5, verbose=1,
                                    refit=True)
model.fit(x_train, y_train, verbose=1,

          eval_set = [(x_train, y_train),(x_test,y_test)],

          eval_metric = 'mlogloss',

          early_stopping_rounds= 50

          )
acc = accuracy_score(y_test, y_pred)
print("Score : ", score)
print("acc : ", acc)
print("최적의 매개변수", model.best_estimator_)
print("최적의 파라메터", model.best_params_)
print("Best Score", model.best_score_)

best_model = model.best_estimator_
aaa = np.sort(best_model.feature_importances_)

r2_list = []
th_list = []

for thresh in aaa:    
    selection = SelectFromModel(best_model, threshold=thresh, prefit= True)
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)    
    # print(select_x_train.shape,select_x_test.shape)
    
    selection_model = XGBClassifier(n_jobs = -1)
    #3 훈련
    selection_model.fit(x_train, y_train, verbose=1,

          eval_set = [(x_train, y_train),(x_test,y_test)],

          eval_metric = 'mlogloss',

          early_stopping_rounds= 50

          )

    #4 평가 예측
    score = selection_model.score(select_x_test, y_test)
    select_y_pred = selection_model.predict(select_x_test)

    select_acc = accuracy_score(y_test, select_y_pred)
    # print("select_Score : ", score)
    print("select_Acc : ", select_acc)    
    # print("Thresh = %.3f, n=%d, R2 :%2f%%" %(thresh,select_x_train.shape[1],select_r2*100))
    r2_list.append(select_acc)
    th_list.append(thresh)
    

index_max_acc = r2_list.index(max(r2_list))
# print(index_max_acc)
drop_list = np.where(model.feature_importances_ < th_list[index_max_acc])
print("제거~!!: ",drop_list)
x,y = load_data(return_X_y=True)
x = np.delete(x,drop_list,axis=1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.8, shuffle = True, random_state = 66
         #, stratify= y
         )

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)