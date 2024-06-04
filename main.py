import pandas as pd 
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from util.preprocess import preprocess
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report 
from sklearn.feature_selection import mutual_info_classif
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.linear_model import LogisticRegression


def oversampling(X_train, Y_train):
    sm = SMOTE(random_state=42)
    X_train, Y_train = sm.fit_resample(X_train, Y_train)
    return X_train, Y_train



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

def model_randomforest(X, Y):
    model = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
    model.fit(X, Y)
    # feature_importances = model.feature_importances_
    # feature_importances = pd.Series(feature_importances)
    # feature_importances.index = X.columns
    # feature_importances = feature_importances.sort_values(ascending=False)
    # print(feature_importances)
    return model
    

def mutual_info(X, Y):
    mi = mutual_info_classif(X, Y)
    mi = pd.Series(mi)
    mi.index = X.columns
    mi = mi.sort_values(ascending=False)
    mi.to_csv('mutual_info.csv')
    print(mi)

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
    X = data.drop(columns=['commit','label'])
    Y = data['label']
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.30, random_state=42)

    X_train, Y_train = oversampling(X_train, Y_train)

    # model = model_randomforest(X_train, Y_train)
    # Y_pred = model.predict(X_test)
    # print(classification_report(Y_test, Y_pred))

    '''Feature selection'''
    model = LinearSVC(C=1, dual = False, random_state=42)
    efs = EFS(model, min_features=1, max_features=12, scoring='f1', cv=5, print_progress=True)
    efs = efs.fit(X_train, Y_train)
    print('Best features:', efs.best_idx_)
    print('Best score:', efs.best_score_)
    print('Best subset:', efs.best_feature_names_)

if __name__ == "__main__":
    main()