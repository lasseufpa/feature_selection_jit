

if __name__ == '__main__':
    import numpy as np
    # import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv('src/features.csv')
    print(len(df.index))
    x1,x2 = [],[]

    for index in range(len(df.index)):
        if df['label'][index] == 1:
            x1.append(df.loc[index])
        else:
            x2.append(df.loc[index])
    print(len(x1))
    print(len(x2))
    print(f'{len(x1)+len(x2)} == {len(df.index)}')
    print(x1[0])