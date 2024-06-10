from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
import pandas as pd


def exhaustive(X_train, Y_train):
    model = LogisticRegression(max_iter=1000,random_state=42)
    smote = SMOTE(random_state=42)
    pipeline = Pipeline([('SMOTE', smote), ('Logistic Regression', model)])
    efs = EFS(pipeline, min_features=1,
            max_features=12,
            scoring='roc_auc',
            cv=5,
            n_jobs=1)
    efs = efs.fit(X_train, Y_train)
    return efs



def main():

    data = pd.read_csv('src/features.csv')
    X = data.drop(columns=['label','commit'])
    Y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    # Defina o modelo
    model = LogisticRegression(max_iter=1000,random_state=42)
    # Defina o método de oversampling
    smote = SMOTE(random_state=42)

    # Crie uma pipeline com o método de oversampling e o modelo
    pipeline = Pipeline([('SMOTE', smote), ('Logistic Regression', model)])

    # Defina o seletor de recursos
    efs = EFS(model, min_features=1,
              max_features=12,
              scoring='roc_auc',
              cv=5,
              n_jobs=1)

    efs = efs.fit(X_train, y_train)

    efs_df = pd.DataFrame.from_dict(efs.get_metric_dict()).T
    efs_df = efs_df.sort_values('avg_score', ascending=False,ignore_index=True)
    efs_df.to_csv('src/Results/LogisticRegression/efs_roc_auc.csv')

if __name__ == '__main__':
    main()
