import pandas as pd 
import os
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def random_forest_all_feature_smote(X,Y):
    model = RandomForestClassifier(n_estimators=500, random_state=42,max_features='log2',max_depth=90,
                               min_samples_split=5, min_samples_leaf=4, criterion='entropy')
    save_metrics(X = X,Y = Y,model = model,smote=True)


def random_forest_all_feature(X,Y):
    model = RandomForestClassifier(n_estimators=500, random_state=42,max_features='log2',max_depth=90,
                               min_samples_split=5, min_samples_leaf=4, criterion='entropy')
    save_metrics(X,Y,model)

def random_forest_top_feature(X,Y, top_features):
    X = X[top_features]
    model = RandomForestClassifier(n_estimators=500, random_state=42,max_features='log2',max_depth=90,
                               min_samples_split=5, min_samples_leaf=4)
    save_metrics(X,Y,model,top_features=top_features)

def random_forest_top_feature_smote(X,Y, top_features):
    X = X[top_features]
    model = RandomForestClassifier(n_estimators=500, random_state=42,max_features='log2',max_depth=90,
                               min_samples_split=5, min_samples_leaf=4, criterion='entropy')
    
    save_metrics(X = X ,Y = Y,model = model,smote = True,top_features = top_features)


def save_metrics(X,Y,model,smote=False,top_features=None):
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
        if os.path.exists(f'src/Results/RandomForest/features_importances/features_importance_{counter}_normalized.csv'):
            counter += 1
            continue
        feature_importance = pipeline.named_steps['randomforestclassifier'].feature_importances_
        feature_importance = pd.DataFrame(feature_importance, index = X.columns, columns=['importance']).sort_values('importance', ascending=False)
        feature_importance.to_csv(f'src/Results/RandomForest/features_importances/features_importance_{counter}_normalized.csv', index =  True)
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
        metrics_df.to_csv('src/Results/RandomForest/metrics_best_features_smote_normalized.csv', index =  False)

    elif top_features:
        metrics_df.to_csv('src/Results/RandomForest/metrics_best_features_normalized.csv', index =  False)

    elif smote:
        metrics_df.to_csv('src/Results/RandomForest/metrics_smote_normalized.csv', index =  False)

    else:
        metrics_df.to_csv('src/Results/RandomForest/metrics_normalized.csv', index =  False)
