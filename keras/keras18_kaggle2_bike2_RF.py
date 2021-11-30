import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

def exam_data_load(df, target, id_name="", null_name=""):
    if id_name == "":
        df = df.reset_index().rename(columns={"index": "id"})
        id_name = 'id'
    else:
        id_name = id_name
    
    if null_name != "":
        df[df == null_name] = np.nan
    
    X_train, X_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=2021)
    y_train = X_train[[id_name, target]]
    X_train = X_train.drop(columns=[id_name, target])
    y_test = X_test[[id_name, target]]
    X_test = X_test.drop(columns=[id_name, target])
    return X_train, X_test, y_train, y_test 

path = "./_data/bike/"
    
df = pd.read_csv(path + "train.csv")
X_train, X_test, y_train, y_test = exam_data_load(df, target='count') #, id_name='Id')
#######################################################################################
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

X_train = X_train.drop(['casual', 'registered'], axis=1)
X_test = X_test.drop(['casual', 'registered'], axis=1)
y_train = y_train.drop('id', axis=1)

print(X_train.head(3))
print(X_train.info())
print(X_train.describe())
print(y_train.head(3))

X_train['datetime'] = pd.to_datetime(X_train['datetime'])
X_test['datetime'] = pd.to_datetime(X_test['datetime'])      

X_train['year'] = X_train['datetime'].dt.year
X_train['month'] = X_train['datetime'].dt.month
X_train['day'] = X_train['datetime'].dt.day
X_train['hour'] = X_train['datetime'].dt.hour
X_train = X_train.drop('datetime', axis=1)

X_test['year'] = X_test['datetime'].dt.year
X_test['month'] = X_test['datetime'].dt.month
X_test['day'] = X_test['datetime'].dt.day
X_test['hour'] = X_test['datetime'].dt.hour
X_test = X_test.drop('datetime', axis=1)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
X_train_model, X_train_val, y_train_model, y_train_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

rf= RandomForestRegressor(random_state=42)
rf.fit(X_train_model, y_train_model)
pred_val = rf.predict(X_train_val)
print('r2 val score:', r2_score(y_train_val, pred_val))

rf.fit(X_train, y_train)
pred = rf.predict(X_test)
submission = pd.DataFrame({'id' : y_test.id, 'count' : pred})
submission.to_csv(path + '000000000.csv', index=False)
print('r2 real score :', r2_score(y_test['count'], pred))

############### 제출용.
# result = model.predict(test_flie)
result = submission['count']
test_flie = pd.read_csv(path + "test.csv") 

submission.drop(['id'], axis=1)
submission['count'] = result
submission['datetime'] = test_flie['datetime']



# print(submission[:10])
submission.to_csv(path+"sampleHR.csv", index = False)