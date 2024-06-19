import pandas as pd
from sklearn.preprocessing import StandardScaler

def normalize_data(data):
    sc = StandardScaler()
    data = sc.fit_transform(data)
    return data

if __name__ == "__main__":
    data = pd.read_csv('src/features.csv')
    data_drop = data.drop(columns=['commit','label'])
    data_drop = normalize_data(data_drop)
    dp = pd.DataFrame(data_drop, columns=data.columns[1:-1])
    dp = pd.merge(left = data['commit'], right = dp, right_index=True, left_index=True)
    dp = pd.merge(left = dp, right = data['label'], right_index=True, left_index=True)
    dp.to_csv('src/normalized_features.csv', index=False)    