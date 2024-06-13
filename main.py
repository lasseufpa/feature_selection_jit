import pandas as pd 
import argparse
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, f1_score, precision_score, recall_score
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold

def pipeline_smote(X,Y):
    model = RandomForestClassifier(n_estimators=500, random_state=42,max_features='log2',max_depth=90,
                               min_samples_split=5, min_samples_leaf=4, criterion='entropy')
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
        # features_importance = pipeline.named_steps['randomforestclassifier'].feature_importances_
        # features_importance = pd.DataFrame(features_importance, index=X.columns, columns=['importance'])
        # features_importance = features_importance.sort_values(by='importance', ascending=False)
        # features_importance.to_csv(f'src/Results/RandomForest/features_importance_{counter}.csv')
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
    
    metrics_df.to_csv('src/Results/RandomForest/metrics_smote.csv', index =  False)


def pipeline_no_smote(X,Y):
    model = RandomForestClassifier(n_estimators=500, random_state=42,max_features='log2',max_depth=90,
                               min_samples_split=5, min_samples_leaf=4, criterion='entropy')
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
    
    metrics_df.to_csv('src/Results/RandomForest/metrics.csv', index =  False)


def cross_validation(X,Y, top_k_features):
    X = X[top_k_features]
    model = RandomForestClassifier(n_estimators=500, random_state=42,max_features='log2',max_depth=90,
                               min_samples_split=5, min_samples_leaf=4)
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
    
    metrics_df.to_csv('src/Results/RandomForest/metrics_best_features.csv', index =  False)


def cross_validation_smote(X,Y, top_k_features):
    X = X[top_k_features]
    smote = SMOTE(random_state=42)
    model = RandomForestClassifier(n_estimators=500, random_state=42,max_features='log2',max_depth=90,
                               min_samples_split=5, min_samples_leaf=4, criterion='entropy')
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
    
    metrics_df.to_csv('src/Results/RandomForest/metrics_best_features_smote.csv', index =  False)


def main():
    parse = argparse.ArgumentParser(description="Dataset of the features")
    parse.add_argument("--data", type=str, required=True, help="Path/to/data.csv")
    args = parse.parse_args()    
    
    try:
        data = pd.read_csv(args.data)
    except:
        print("File not found")
        args.data = input("Digite o caminho do arquivo: ")
        data = pd.read_csv(args.data)
    
    X = data.drop(columns=['commit','label'])
    Y = data['label']
    ranking = pd.read_json('src/Results/RandomForest/ranking_medio.json')
    top_k_features = ranking[:4]
    top_k_features = top_k_features[0].to_list()
    pipeline_smote(X,Y)
    pipeline_no_smote(X,Y)
    cross_validation(X,Y, top_k_features)
    cross_validation_smote(X,Y, top_k_features)


if __name__ == "__main__":
    main()