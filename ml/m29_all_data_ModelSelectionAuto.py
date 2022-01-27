import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_diabetes, load_boston, fetch_covtype
from xgboost import XGBRegressor, XGBClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings(action='ignore')

#GridSerch 적용해서 출력한 값에서 
#피쳐임포턴스 출력 후
#SelectModel 만들어서
#칼럼 축소 후
#모델 구축하여 결과 도출
#diabet, boston, fetchcov,winequality

datasets = {'Diabets':load_diabetes(),'Boston':load_boston(),'WineQuality':'WineQuality'}#,'FetchCov':fetch_covtype()}

reg_param_grid={'booster' :['gbtree'],            
                #  'silent':[True],
                 'max_depth':range (10, 20, 5),
                 'min_child_weight':[5],
                 'gamma':[1],
                #  'nthread':[4],
                 'colsample_bytree':[0.5],
                 'colsample_bylevel':[0.9],
                 'n_estimators':[100000],
                #  'objective':['binary:logistic'],
                 'learning_rate':np.arange(0.001,0.005,0.001),
                 'random_state':[2]}

cl_param_grid={'booster' :['gbtree'],            
                #  'silent':[True],
                 'max_depth':range (10, 20, 5),
                 'min_child_weight':[5],
                 'gamma':[1],
                #  'nthread':[4],
                 'colsample_bytree':[0.5],
                 'colsample_bylevel':[0.9],
                 'n_estimators':[10000000],
                 'objective':['mlogloss'],
                 'learning_rate':np.arange(0.001,0.005,0.001),
                 'random_state':[2]}
n_job = -1

f = open('AllModelSelection.txt','w')
for (dataset_name, dataset) in datasets.items():    
    print(f"-----------{dataset_name}--------------")    
    f.write(f"-----------{dataset_name}--------------\n")
    
    if dataset_name == 'Boston' or 'Diabets':
        x = dataset.data
        y = dataset.target
        
        x_train, x_test, y_train, y_test = train_test_split(x, y,
                train_size = 0.8, shuffle = True, random_state = 66)
        
        scaler = StandardScaler()
        scaler.fit_transform(x_train)
        scaler.transform(x_test)        
        
        model = GridSearchCV(XGBRegressor(),reg_param_grid, cv=5, verbose=1,
                     refit=True, n_jobs=n_job)
        # 훈련
        model.fit(x_train, y_train, verbose=1,

            eval_set = [(x_train, y_train),(x_test,y_test)],

            eval_metric = 'rmse',

            early_stopping_rounds= 50

            )
    else: 
        path = "D:\\_data\\"
        dataset = pd.read_csv(path + "winequality-white.csv",index_col=None, header=0, sep=';')   
        
        x = dataset.drop('quality', axis=1)
        y = dataset.quality   
        
        x_train, x_test, y_train, y_test = train_test_split(x, y,
                train_size = 0.8, shuffle = True, random_state = 66)
        
        scaler = StandardScaler()
        scaler.fit_transform(x_train)
        scaler.transform(x_test)
         
        model = GridSearchCV(XGBClassifier(), cl_param_grid, cv=5, verbose=1,
                     refit=True, n_jobs=n_job)
        
        # 훈련
        model.fit(x_train, y_train, verbose=1,

            eval_set = [(x_train, y_train),(x_test,y_test)],

            eval_metric = 'mlogloss',

            early_stopping_rounds= 50

          )
    
    start = time.time()
    
    # 평가 예측
    result = model.score(x_test, y_test)
    
    # #4 평가 예측    
    score = model.score(x_test, y_test)
    y_pred = model.predict(x_test)

    # acc = accuracy_score(y_test, y_pred)
    print("Score : ", score)
    f.write(f"Score : {score} \n")
    # print("acc : ", acc)
    print("최적의 매개변수", model.best_estimator_)
    f.write(f"최적의 매개변수 {model.best_estimator_,}\n")
    
    print("최적의 파라메터", model.best_params_)
    f.write(f"최적의 파라메터 {model.best_params_}\n")
    
    print("Best Score", model.best_score_)
    f.write(f"Best Score {model.best_score_}\n")
    

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
    selection_model.fit(select_x_train, y_train)

    #4 평가 예측
    score = selection_model.score(select_x_test, y_test)
    select_y_pred = selection_model.predict(select_x_test)

    # select_acc = accuracy_score(y_test, select_y_pred)
    # print("select_Score : ", score)
    print(select_x_train.shape[1], " select_Acc : ", score)    
    # print("Thresh = %.3f, n=%d, R2 :%2f%%" %(thresh,select_x_train.shape[1],select_r2*100))
    r2_list.append(score)
    th_list.append(thresh)
    

index_max_acc = r2_list.index(max(r2_list))
drop_list = np.where(model.feature_importances_ < th_list[index_max_acc])
print("제거~!!: ",drop_list)


if dataset_name == 'Boston' or 'Diabets':
    x = dataset.data
    y = dataset.target
    
    x = np.delete(x,drop_list,axis=1)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y,
            train_size = 0.8, shuffle = True, random_state = 66)
    
    scaler = StandardScaler()
    scaler.fit_transform(x_train)
    scaler.transform(x_test) 
    
    #2 모델
    model = XGBRegressor(n_jobs = -1)
    #3 훈련
    model.fit(x_train, y_train, verbose=1,

          eval_set = [(x_train, y_train),(x_test,y_test)],

          eval_metric = 'rmse',

          early_stopping_rounds= 50

          )

    #4 평가 예측
    score = model.score(x_test, y_test)
    y_pred = model.predict(x_test)
    print('================================= 수정 후')
    f.write('================================= 수정 후\n')
    
    # r2 = r2_score(y_test, y_pred)
    print("Score : ", score)
    f.write(f"Score : {score}\n")
    
    # print("r2 : ", r2)
    print('=================================')

else: 
    path = "D:\\_data\\"
    dataset = pd.read_csv(path + "winequality-white.csv",index_col=None, header=0, sep=';')   
    
    x = dataset.drop('quality', axis=1)
    y = dataset.quality   
    
    x_train, x_test, y_train, y_test = train_test_split(x, y,
            train_size = 0.8, shuffle = True, random_state = 66)
    
    scaler = StandardScaler()
    scaler.fit_transform(x_train)
    scaler.transform(x_test)
    
    #2 모델
    model = XGBClassifier(n_jobs = -1)
    #3 훈련
    model.fit(x_train, y_train, verbose=1,

          eval_set = [(x_train, y_train),(x_test,y_test)],

          eval_metric = 'mlogloss',

          early_stopping_rounds= 50

          )

    #4 평가 예측
    score = model.score(x_test, y_test)
    y_pred = model.predict(x_test)
    print('================================= 수정 후')
    f.write('================================= 수정 후\n')
    # r2 = r2_score(y_test, y_pred)
    print("Score : ", score)
    f.write(f"Score : {score}\n")
    # print("r2 : ", r2)
    print('=================================')
    
f.close()




