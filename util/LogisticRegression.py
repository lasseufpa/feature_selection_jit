import pandas as pd 
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold


def logistic_regression_all_feature(X,Y):
    model = LogisticRegression(max_iter=1000,random_state=42)
    cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    metrics_df = pd.DataFrame(columns=['Fold', 'Precision', 'Recall', 'F1-score', 'Roc_auc_score'])
    counter = 1
    for train_index, test_index in cv.split(X, Y):
        X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
        Y_train_cv, Y_test_cv = Y.iloc[train_index], Y.iloc[test_index]
        model.fit(X_train_cv, Y_train_cv)
        y_pred = model.predict(X_test_cv)
        metrics_df = metrics_df._append({'Fold': counter,
                        'Precision': precision_score(Y_test_cv, y_pred),
                        'Recall':recall_score(Y_test_cv, y_pred),
                        'F1-score':f1_score(Y_test_cv, y_pred),
                        'Roc_auc_score':roc_auc_score(Y_test_cv,y_pred)}, ignore_index=True)
        counter += 1
    metrics_df = metrics_df._append({'Fold': 'Mean',
                                     'Precision': metrics_df['Precision'].mean(),
                                     'Recall': metrics_df['Recall'].mean(),
                                     'F1-score': metrics_df['F1-score'].mean(),
                                     'Roc_auc_score': metrics_df['Roc_auc_score'].mean()}, ignore_index=True)
    metrics_df = metrics_df._append({'Fold': 'Std',
                                     'Precision': metrics_df['Precision'].std(),
                                     'Recall': metrics_df['Recall'].std(),
                                     'F1-score': metrics_df['F1-score'].std(),
                                     'Roc_auc_score': metrics_df['Roc_auc_score'].std()}, ignore_index=True)
    metrics_df.to_csv('src/Results/LogisticRegression/metrics.csv', index =  False)

def logistic_regression_all_feature_smote(X,Y):
    model = LogisticRegression(max_iter=1000,random_state=42)
    smote = SMOTE(random_state=42)
    pipeline = make_pipeline(smote, model)
    cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    metrics_df = pd.DataFrame(columns=['Fold', 'Precision', 'Recall', 'F1-score', 'Roc_auc_score'])
    counter = 1
    for train_index, test_index in cv.split(X, Y):
        X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
        Y_train_cv, Y_test_cv = Y.iloc[train_index], Y.iloc[test_index]
        pipeline.fit(X_train_cv, Y_train_cv)
        y_pred = pipeline.predict(X_test_cv)
        metrics_df = metrics_df._append({'Fold': counter,
                        'Precision': precision_score(Y_test_cv, y_pred),
                        'Recall':recall_score(Y_test_cv, y_pred),
                        'F1-score':f1_score(Y_test_cv, y_pred),
                        'Roc_auc_score':roc_auc_score(Y_test_cv,y_pred)}, ignore_index=True)
        counter += 1
    metrics_df = metrics_df._append({'Fold': 'Mean',
                                     'Precision': metrics_df['Precision'].mean(),
                                     'Recall': metrics_df['Recall'].mean(),
                                     'F1-score': metrics_df['F1-score'].mean(),
                                     'Roc_auc_score': metrics_df['Roc_auc_score'].mean()}, ignore_index=True)
    metrics_df = metrics_df._append({'Fold': 'Std',
                                     'Precision': metrics_df['Precision'].std(),
                                     'Recall': metrics_df['Recall'].std(),
                                     'F1-score': metrics_df['F1-score'].std(),
                                     'Roc_auc_score': metrics_df['Roc_auc_score'].std()}, ignore_index=True)
    metrics_df.to_csv('src/Results/LogisticRegression/metrics_smote.csv', index =  False)

def logistic_regression_top_feature(X,Y, top_features):
    X=X[top_features]
    model = LogisticRegression(max_iter=1000,random_state=42)
    cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    metrics_df = pd.DataFrame(columns=['Fold', 'Precision', 'Recall', 'F1-score', 'Roc_auc_score'])

    counter = 1
    for train_index, test_index in cv.split(X, Y):
        X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
        Y_train_cv, Y_test_cv = Y.iloc[train_index], Y.iloc[test_index]
        model.fit(X_train_cv, Y_train_cv)
        y_pred = model.predict(X_test_cv)
        metrics_df = metrics_df._append({'Fold': counter,
                                         'Precision': precision_score(Y_test_cv, y_pred),
                                         'Recall':recall_score(Y_test_cv, y_pred),
                                         'F1-score':f1_score(Y_test_cv, y_pred),
                                         'Roc_auc_score':roc_auc_score(Y_test_cv,y_pred)}, ignore_index=True)
        counter += 1
    metrics_df = metrics_df._append({'Fold': 'Mean',
                                     'Precision': metrics_df['Precision'].mean(),
                                     'Recall': metrics_df['Recall'].mean(),
                                     'F1-score': metrics_df['F1-score'].mean(),
                                     'Roc_auc_score': metrics_df['Roc_auc_score'].mean()}, ignore_index=True)
    metrics_df = metrics_df._append({'Fold': 'Std',
                                     'Precision': metrics_df['Precision'].std(),
                                     'Recall': metrics_df['Recall'].std(),
                                     'F1-score': metrics_df['F1-score'].std(),
                                     'Roc_auc_score': metrics_df['Roc_auc_score'].std()}, ignore_index=True)

    metrics_df.to_csv('src/Results/LogisticRegression/metrics_best_features.csv', index =  False)

def logistic_regression_top_feature_smote(X,Y, top_features):
    X=X[top_features]
    model = LogisticRegression(max_iter=1000,random_state=42)
    smote = SMOTE(random_state=42)
    cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    pipeline = make_pipeline(smote, model)
    metrics_df = pd.DataFrame(columns=['Fold', 'Precision', 'Recall', 'F1-score', 'Roc_auc_score'])

    counter = 1
    for train_index, test_index in cv.split(X, Y):
        X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
        Y_train_cv, Y_test_cv = Y.iloc[train_index], Y.iloc[test_index]
        pipeline.fit(X_train_cv, Y_train_cv)
        y_pred = model.predict(X_test_cv)
        metrics_df = metrics_df._append({'Fold': counter,
                                         'Precision': precision_score(Y_test_cv, y_pred),
                                         'Recall':recall_score(Y_test_cv, y_pred),
                                         'F1-score':f1_score(Y_test_cv, y_pred),
                                         'Roc_auc_score':roc_auc_score(Y_test_cv,y_pred)}, ignore_index=True)
        counter += 1
    metrics_df = metrics_df._append({'Fold': 'Mean',
                                     'Precision': metrics_df['Precision'].mean(),
                                     'Recall': metrics_df['Recall'].mean(),
                                     'F1-score': metrics_df['F1-score'].mean(),
                                     'Roc_auc_score': metrics_df['Roc_auc_score'].mean()}, ignore_index=True)
    metrics_df = metrics_df._append({'Fold': 'Std',
                                     'Precision': metrics_df['Precision'].std(),
                                     'Recall': metrics_df['Recall'].std(),
                                     'F1-score': metrics_df['F1-score'].std(),
                                     'Roc_auc_score': metrics_df['Roc_auc_score'].std()}, ignore_index=True)

    metrics_df.to_csv('src/Results/LogisticRegression/metrics_best_features_smote.csv', index =  False)

