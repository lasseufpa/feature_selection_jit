import pandas as pd 
import argparse 

def preprocess(data):
    data = data.dropna()
    data = data.drop_duplicates()
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path/to/data.csv")
    args = parser.parse_args()
    try:
        data = pd.read_csv(args.data)
    except:
        print("File not found")
        return
    data = preprocess(data)
    data.to_csv('features.csv', index=False)


if __name__ == "__main__":
    main()
