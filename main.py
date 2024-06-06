import pandas as pd 
import argparse
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from util.preprocess import preprocess
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.feature_selection import mutual_info_classif
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.linear_model import LogisticRegression


def oversampling(X, Y):
    sm = SMOTE(random_state=42)
    X, Y = sm.fit_resample(X, Y)
    return X, Y

def undersampling(X, Y):
    tl = TomekLinks()
    X, Y = tl.fit_resample(X, Y)
    return X, Y

def combine_sampling(X, Y):
    smt = SMOTETomek(random_state=42)
    X, Y = smt.fit_resample(X, Y)
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
    # mi.to_csv('mutual_info.csv')
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
    # X =  data[['number_unique_changes','lines_of_code_added']]
    Y = data['label']
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.30, random_state=42)

    '''Oversampling the data'''
    X_train, Y_train = oversampling(X_train, Y_train)
    # model = model_randomforest(X_over, Y_over)
    # Y_pred = model.predict(X_test)
    # print('\nClassification report Random Forest (Oversampling):\n')
    # print(classification_report(Y_test, Y_pred))

    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model = model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    print('\nClassification report Logistic Regression:\n')
    print(classification_report(Y_test, y_pred))
    print(f'\nROC AUC Logistic Regression: {roc_auc_score(y_score=y_pred,y_true=Y_test)}\n')


if __name__ == "__main__":
    main()