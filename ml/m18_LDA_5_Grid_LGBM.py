import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import time

import warnings
warnings.filterwarnings(action='ignore')

(x_train, y_train),(x_test,y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])

scale = StandardScaler()
scale.fit_transform(x_train)
scale.transform(x_test)

n_features = x_train.shape[1]    
clases = len(np.unique(y_train))        
n_component = min(clases-1, n_features)

print("LDA n_component = ",n_component)
lda = LinearDiscriminantAnalysis(n_components=n_component)  
      
lda.fit(x_train,y_train)
x_train = lda.transform(x_train)
x_test = lda.transform(x_test) 

# model = XGBClassifier()

xg_parameters ={'max_depth' : [3,4,5,6] , 'n_estimators': [12,24,32], 
                'learning_rate':[0.01, 0.1], 'gamma': [0.5, 1, 2], 'random_state':[99]}

start = time.time()

from lightgbm import LGBMClassifier
lgbm = LGBMClassifier(n_estimators=200)

lgb_params = {'max_depth': [10, 15, 20],
          'min_child_samples': [20, 40, 60],
          'subsample': [0.8, 1]}
n_job = -1
# model = GridSearchCV(XGBClassifier(), xg_parameters, cv=5, verbose=1 ,refit=True, n_jobs=n_job)
# model.fit(x_train,y_train, eval_metric = 'merror')
model = GridSearchCV(LGBMClassifier(), lgb_params, cv=5, verbose=1 ,refit=True, n_jobs=n_job)
model.fit(x_train,y_train)

result = model.score(x_test,y_test)
print("GridSearchCV LGBMClassifier LDA score :",result)

print("최적 파라미터: ", model.best_params_) # {'max_depth': 15, 'min_child_samples': 40, 'subsample': 0.8}
print("걸린시간 ", time.time() - start)
'''
GridSearchCV LGBMClassifier LDA score : 0.9162
'''