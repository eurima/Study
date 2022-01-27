import matplotlib.pyplot as plt
from matplotlib import axis
from sklearn.datasets import load_iris as load_data
import numpy as np
import pandas as pd
import time
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings(action='ignore')
dataset = load_data()

x = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)
# 머신러닝 모델 구성
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from xgboost.plotting import plot_importance


model_3 = XGBClassifier(random_state=66)
model_3.fit(x_train,y_train)


plot_importance(model_3)
plt.show()
