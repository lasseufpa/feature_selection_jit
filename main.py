import pandas as pd 
import argparse
from util.RandomForest import random_forest_all_feature, random_forest_all_feature_smote, random_forest_top_feature, random_forest_top_feature_smote    


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
    random_forest_all_feature_smote(X,Y)

if __name__ == "__main__":
    main()

    