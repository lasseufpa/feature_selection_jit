import pandas as pd 
import os
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold

def support_vector_machine_all_feature_smote(X,Y):
    model = SVC(C=0.01,gamma=1e-09)
    save_metrics(X = X,Y = Y,model = model,smote=True)


def support_vector_machine_all_feature(X,Y):
    model = SVC(C=0.01,gamma=1e-09)
    save_metrics(X,Y,model)


def support_vector_machine_top_feature(X,Y, top_features):
    model = SVC(C=0.01,gamma=1e-09)
    save_metrics(X,Y,model,top_features=top_features)

def support_vector_machine_top_feature_smote(X,Y, top_features):
    model = SVC(C=0.01,gamma=1e-09)
    save_metrics(X = X,Y = Y,model = model,smote = True,top_features = top_features)


def save_metrics(X,Y,model,smote = False, top_features =  None):
    if top_features:
        X = X[top_features]
    if smote:
        smote = SMOTE(random_state=42)
        pipeline = make_pipeline(smote, model)
    else:
        pipeline = model
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
    
    if smote and top_features:
        metrics_df.to_csv('src/Results/SupportVectorMachine/metrics_best_features_smote.csv', index =  False)

    elif top_features:
        metrics_df.to_csv('src/Results/SupportVectorMachine/metrics_best_features.csv', index =  False)

    elif smote:
        metrics_df.to_csv('src/Results/SupportVectorMachine/metrics_smote.csv', index =  False)

    else:
        metrics_df.to_csv('src/Results/SupportVectorMachine/metrics.csv', index =  False)