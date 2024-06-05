from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
import pandas as pd

data = pd.read_csv('src/features.csv')
X = data.drop(columns=['label','commit'])
Y = data['label']


# Divida o dataset em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Defina o modelo
model = LogisticRegression()

# Defina o método de oversampling
smote = SMOTE()

# Crie uma pipeline com o método de oversampling e o modelo
pipeline = Pipeline([('SMOTE', smote), ('Logistic Regression', model)])

# Defina o seletor de recursos
efs = EFS(pipeline, min_features=1, max_features=2, scoring='roc_auc', cv=5)

# Ajuste o seletor de recursos aos dados de treinamento
efs = efs.fit(X_train, y_train)

# Imprima as melhores características encontradas
print('Best features:', efs.best_idx_)
print('Best features names:', efs.best_feature_names_)

# Avalie o desempenho no conjunto de teste usando as melhores características (opcional)
X_train_selected = X_train.iloc[:, list(efs.best_idx_)]
X_test_selected = X_test.iloc[:, list(efs.best_idx_)]
pipeline.fit(X_train_selected, y_train)
test_score = pipeline.score(X_test_selected, y_test)
print('Test score (roc_auc):', test_score)
