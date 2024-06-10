import pandas as pd
import numpy as np

data = pd.read_csv('src/Results/RandomForest/grid_search_results_random_forest.csv')
data.sort_values(by='rank_test_score', inplace=True)
data.drop(columns=['mean_fit_time','std_fit_time','std_score_time',
                   'split0_test_score','split1_test_score','split2_test_score',
                   'split3_test_score','split4_test_score'],inplace=True)
for i in range(len(data.index)):
    if data['rank_test_score'][i] != 1:
        data.drop(index=i,inplace=True)
data.reset_index(inplace=True)

for i in range(len(data.index)):
    print(data['params'][i])