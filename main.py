import pandas as pd 
import argparse
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from util.preprocess import preprocess
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report 
from sklearn.feature_selection import mutual_info_classif
from mlxtend.feature_selection import exhaustive_feature_selector as EFS
from sklearn.linear_model import LogisticRegression


def oversampling(X_train, Y_train):
    sm = SMOTE(random_state=42)
    X_train, Y_train = sm.fit_resample(X_train, Y_train)
    return X_train, Y_train

def undersampling(X, Y):
    nm = NearMiss(version=1)
    X, Y = nm.fit_resample(X, Y)
    return X, Y


def plot_correlation_matrix(data):
    plt.figure(figsize=(10,10))
    mask = np.triu(np.ones_like(data.corr(), dtype=bool), 1)
    sns.heatmap(data.corr(),
                annot=True,
                fmt='.2f',
                cmap='Blues',
                mask=mask,
                cbar=True)
    plt.tight_layout()
    plt.savefig('correlacao_features.png', format='png', bbox_inches='tight')
    # plt.show()


def exhaustive_feature_selector(X_train, Y_train):
    model = LogisticRegression()
    efs = EFS(model, min_features=1, max_features=12, scoring='accuracy', cv=5)
    efs = efs.fit(X_train, Y_train)
    print('Best accuracy score: %.2f' % efs.best_score_)
    print('Best subset (indices):', efs.best_idx_)
    print('Best subset (corresponding names):', efs.best_feature_names_)

def model_randomforest(X, Y):
    model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X, Y)
    feature_importances = model.feature_importances_
    feature_importances = pd.Series(feature_importances)
    feature_importances.index = X.columns
    feature_importances = feature_importances.sort_values(ascending=False)
    print(feature_importances)
    return model
    

def mutual_info(X, Y):
    mi = mutual_info_classif(X, Y)
    mi = pd.Series(mi)
    mi.index = X.columns
    mi = mi.sort_values(ascending=False)
    mi.to_csv('mutual_info.csv')
    print(mi)


def filter_rows(df, column, value):
    return df.loc[df[column] == value]

def main():
    parse = argparse.ArgumentParser(description="Dataset of the features")
    parse.add_argument("--data", type=str, required=True, help="Path/to/data.csv")
    args = parse.parse_args()    
    try:
        data = pd.read_csv(args.data)
    except:
        print("File not found")
        return 
    data = preprocess(data)
    
    '''Splitting the data into training and testing sets.'''
    # X =  data.drop(columns=['label','commit'])
    
    # data = filter_rows(data,'label',0)
    X = data.drop(columns=['commit','label'])
    Y = data['label']
    # plot_correlation_matrix(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.30, random_state=42)
    print("Classification report for the original dataset")
    
    model2 = model_randomforest(X_train, Y_train)
    y_pred2 = model2.predict(X_test)
    print(classification_report(Y_test, y_pred2))

    X_over, Y_over = oversampling(X_train,Y_train)
    print("Classification report for the oversampled dataset")

    model = model_randomforest(X_over, Y_over)
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred))
    
    print("Classification report for the undersampled dataset")  
    X_under, Y_under = undersampling(X_train,Y_train)
    model = model_randomforest(X_under, Y_under)
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred))
    
if __name__ == "__main__":
    main()