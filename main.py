import pandas as pd 
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from util.preprocess import preprocess
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report 


def oversampling(X_train, Y_train):
    sm = SMOTE(random_state=42)
    X_train, Y_train = sm.fit_resample(X_train, Y_train)
    return X_train, Y_train


def scaling(X_train):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    return X_train


def plot_correlation_matrix(data):
    plt.figure(figsize=(12, 10))
    plt.rcParams['xtick.labelsize']=13
    plt.rcParams['ytick.labelsize']=13
    plt.rcParams.update({'font.size': 13})

    sns.heatmap(data.corr(),
                annot=True,
                fmt='.2f',
                cmap='Blues')
    plt.title('Correlação entre as features com oversampling SMOTE',fontsize = 14)
    plt.tight_layout()
    plt.show()



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

    '''Splitting the data into training and testing sets and applying SMOTE to balance the data.'''
    X =  data.drop(columns=['label','commit'])
    Y = data['label']
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3, random_state=42)

    # X_train, Y_train = oversampling(X_train, Y_train)
    # X_train = scaling(X_train)
    # plot_correlation_matrix(data)

    '''Training the model'''
    model = RandomForestClassifier()
    model.fit(X_train, Y_train)

    importances = model.feature_importances_
    features = X_train.columns
    importance_df = pd.DataFrame({'feature': features, 'importance': importances})
    importance_df = importance_df.sort_values('importance', ascending=False)
    print(importance_df)

    # y_pred = model.predict(X_test)
    # report = classification_report(Y_test, y_pred)
    # print(report)

    
if __name__ == "__main__":
    main()