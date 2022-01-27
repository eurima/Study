import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_covtype
import warnings
import os
warnings.filterwarnings(action='ignore')
datasets = fetch_covtype()
x = datasets.data
y = datasets.target 
    
report_path = 'D:\Study\_Report\\'#'D\\Study\_Report\\
file_name = os.path.abspath( __file__ ).split('\\')[3].split('.')[0]
file = open(report_path + file_name +".txt", "w")

model = XGBClassifier(n_jobs = -1)
x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size = 0.8, shuffle = True, random_state = 66
        , stratify= y)
msg = '========== SMOTE 적용 전 ==============\n'
print(msg)  
file.write(msg)
msg = pd.Series(y).value_counts()
print(msg)  
file.write(f'{msg}\n')   

msg ='========== SMOTE 적용 ==============\n'   
print(msg)  
file.write(msg)

smote = SMOTE(random_state=66)
x_train, y_train = smote.fit_resample(x_train, y_train)

msg = pd.Series(y_train).value_counts()
print(msg)  
file.write(f'{msg}\n')  

np.save('fetch_covtype_datax_x_train',x_train)
np.save('fetch_covtype_datax_y_train',y_train)
np.save('fetch_covtype_datax_x_test',x_test)
np.save('fetch_covtype_datax_y_test',y_test)

    
file.close()
'''

'''