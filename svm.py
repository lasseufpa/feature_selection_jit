import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(0, 0.5 * (self.midpoint - self.vmin) / (self.midpoint - self.vmin))
        normalized_max = min(1, 0.5 * (self.vmax - self.midpoint) / (self.vmax - self.midpoint) + 0.5)
        normalized_value = np.ma.masked_array(np.interp(value, [self.vmin, self.midpoint, self.vmax], [0, normalized_min, normalized_max]))
        return normalized_value


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
    "Os melhores paramentros são: %s com uma pontuação de ROC-AUC:%0.2f"
    % (grid.best_params_, grid.best_score_)
)
print(f'Best Roc Auc: {grid.best_score_}')


'''Vizualização dos Parametros'''

# Obtenha os resultados do GridSearchCV
results = grid.cv_results_

# Extraia as pontuações de ROC AUC e os parâmetros correspondentes
mean_test_scores = results['mean_test_score']
param_C = results['param_SVM__C'].data
param_gamma = results['param_SVM__gamma'].data

# Crie uma matriz para armazenar as pontuações
scores_matrix = mean_test_scores.reshape(len(C_range), len(gamma_range))

# Plotagem
plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=0.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(
    scores_matrix,
    interpolation="nearest",
    cmap=plt.cm.hot,
    norm=MidpointNormalize(vmin=0.2, midpoint=0.92)
)
plt.xlabel("gamma")
plt.ylabel("C")
plt.colorbar()
plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
plt.yticks(np.arange(len(C_range)), C_range)
plt.title("ROC AUC Scores for different C and gamma values")
plt.savefig('roc_auc_heatmap.pdf', format='pdf')
plt.close()
