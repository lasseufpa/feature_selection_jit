import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


# Função de normalização de cores personalizada para centralizar a cor na média das pontuações
class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(0, 0.5 * (self.midpoint - self.vmin) / (self.midpoint - self.vmin))
        normalized_max = min(1, 0.5 * (self.vmax - self.midpoint) / (self.vmax - self.midpoint) + 0.5)
        normalized_value = np.ma.masked_array(np.interp(value, [self.vmin, self.midpoint, self.vmax], [0, normalized_min, normalized_max]))
        return normalized_value

# Carregar os dados
data = pd.read_csv('src/features.csv')
X = data.drop(columns=['label', 'commit'])
Y = data['label']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Definir SMOTE e modelo Random Forest
smote = SMOTE(random_state=42)
model = RandomForestClassifier(random_state=42)

# Criar pipeline
pipeline = Pipeline([('SMOTE', smote), ('RF', model)])

# Definir intervalo de parâmetros para GridSearch
param_grid = {
    'RF__n_estimators': [100, 200, 300, 400, 500],
    'RF__max_features': ['sqrt', 'log2'],
    'RF__max_depth': [None, 10, 30, 50, 70, 90, 110],
    'RF__min_samples_split': [2, 5, 10],
    'RF__min_samples_leaf': [1, 2, 4]
}

# Configurar validação cruzada
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)

# Configurar e executar GridSearch
grid = GridSearchCV(pipeline, param_grid=param_grid, cv=cv, scoring='roc_auc', n_jobs=1)
grid.fit(X_train, Y_train)

# Verificar os melhores parâmetros encontrados
print("Best parameters found: ", grid.best_params_)
print("Best ROC AUC score: ", grid.best_score_)

# Obtenha os resultados do GridSearchCV
results = grid.cv_results_
results_df = pd.DataFrame(results)
results_df.to_csv('grid_search_results_random_forest.csv')
