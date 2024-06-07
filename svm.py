import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.svm import SVC


data = pd.read_csv('src/features.csv')
X = data.drop(columns=['label','commit'])
Y = data['label']

smote = SMOTE(random_state=42)
model = SVC()

pipeline = Pipeline([('SMOTE', smote), ('SVM', model)])

C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(SVC__gamma=gamma_range, SVC__C=C_range)

cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
grid = GridSearchCV(pipeline, param_grid=param_grid, cv=cv, scoring='roc_auc', n_jobs=1)
grid.fit(X, Y)

print(
    "The best parameters are %s with a score of %0.2f"
    % (grid.best_params_, grid.best_score_)
)
print(f'Best Roc Auc: {grid.best_score_}')


'''Vizualização dos Parametros'''
