import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.svm import SVC


# Carregar os dados
data = pd.read_csv('src/features.csv')
X = data.drop(columns=['label', 'commit'])
Y = data['label']

smote = SMOTE(random_state=42)
model = SVC()


pipeline = Pipeline([('SMOTE', smote), ('SVM', model)])
C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = {'SVM__gamma': gamma_range, 'SVM__C': C_range}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = RandomizedSearchCV(pipeline, param_grid, cv=cv, scoring='roc_auc', n_jobs=1)
grid.fit(X, Y)

# Obtenha os resultados do GridSearchCV
results = grid.cv_results_
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by=['rank_test_score'])
results_df.to_csv('grid_search_results_svm.csv', index=False)
