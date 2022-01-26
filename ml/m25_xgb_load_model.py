from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=66)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from xgboost import XGBRegressor
model = XGBRegressor()
path = './_save/'
model.load_model(path + 'xgb_model.dat')

score = model.score(x_test,y_test)
results = model.evals_result()
print(score)

