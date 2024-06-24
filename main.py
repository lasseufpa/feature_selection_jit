import pandas as pd 
import argparse 
from util.SupportVectorMachine import support_vector_machine_all_feature, support_vector_machine_all_feature_smote
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE

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
    # random_forest_all_feature_smote(X,Y)
    # random_forest_all_feature(X,Y)
    model = SVC(C=0.01,gamma=1e-09)
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=42)
    smote = SMOTE(random_state=42)
    X_train, Y_train = smote.fit_resample(X_train,Y_train)
    model.fit(X_train,Y_train)
    y_pred = model.predict(X_test)
    print(classification_report(Y_test,y_pred))
    # print(f'Precision: {precision_score(Y_test, y_pred)}')
    # print(f'Recall: {recall_score(Y_test, y_pred)}')
    # print(f'F1-score: {f1_score(Y_test, y_pred)}')
    # print(f'Roc_auc_score: {roc_auc_score(Y_test,y_pred)}')

    # top_features = ['number_unique_changes','lines_of_code_added','files_churned','number_of_authors']
    # support_vector_machine_all_feature(X,Y)
    # support_vector_machine_all_feature_smote(X,Y)
    

if __name__ == "__main__":
    main()

    