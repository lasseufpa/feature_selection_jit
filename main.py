import pandas as pd 
import argparse
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from util.preprocess import preprocess
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.metrics import roc_auc_score, classification_report


def oversampling(X, Y):
    sm = SMOTE(random_state=42)
    X, Y = sm.fit_resample(X, Y)
    return X, Y

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
        args.data = input("Digite o caminho correto do arquivo: ")
        data = pd.read_csv(args.data)

    data = preprocess(data)
    
    '''Splitting the data into training and testing sets.'''

    X = data.drop(columns=['commit','label'])
    Y = data['label']
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.30, random_state=42)
    X_train, Y_train = oversampling(X_train, Y_train)
    
    model = RandomForestClassifier(n_estimators=500, random_state=42,max_features='log2',max_depth=90,
                                   min_samples_split=5, min_samples_leaf=4)
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    print(f'\n\n Resultado usando todas as features:\n')
    print(f'{classification_report(Y_test, y_pred, tagert_names = ['Não introduz Bug', 'Introduz Bug'],)}')
    print(f'Score:{roc_auc_score(Y_test, y_pred)}')

    X = data[['number_unique_changes','lines_of_code_added','files_churned','entropy']]
    Y = data['label']
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.30, random_state=42)
    X_train, Y_train = oversampling(X_train, Y_train)
    model = RandomForestClassifier(n_estimators=500, random_state=42,max_features='log2',max_depth=90,
                                   min_samples_split=5, min_samples_leaf=4)
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    print(f'\n\nResultado usando as 4 features mais importantes: \n')
    print(f'Score:{roc_auc_score(Y_test, y_pred)}')


if __name__ == "__main__":
    main()