import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


data = pd.read_csv('src/features.csv')
X = data.drop(columns=['label','commit'])
Y = data['label']
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42, n_jobs=1)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv, scoring='roc_auc', n_jobs=1)
grid.fit(X, Y)

print(
    "The best parameters are %s with a score of %0.2f"
    % (grid.best_params_, grid.best_score_)
)

'''Vizualização dos Parametros'''
