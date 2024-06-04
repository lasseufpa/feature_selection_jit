import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# Generate synthetic imbalanced dataset
# X, Y = make_classification(n_samples=10000, n_features=12, n_informative=2, n_redundant=0,
#                            n_clusters_per_class=1, weights=[0.9, 0.1], flip_y=0, random_state=1)

data = pd.read_csv('src/features.csv')
X = data.drop(columns=['commit','label'],axis=1).values
Y = data['label'].values

'''
minority_class_label = 1  # replace with the label of your minority class
X_minority = X[Y == minority_class_label]

# Fit the KDE model on the minority class
kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X_minority)

# Generate new samples from the KDE
num_new_samples = len(X_minority)
X_new_samples = kde.sample(num_new_samples)

# Combine new samples with the original data
X_resampled = np.vstack([X, X_new_samples])
y_resampled = np.hstack([Y, np.ones(num_new_samples)])
'''
# Plot the original and resampled data
plt.scatter(X[Y == 0][:, 1], X[Y == 0][:, -1], label='Majority class', alpha=0.5)
plt.scatter(X[Y == 1][:, 1], X[Y == 1][:, -1], label='Minority class', alpha=0.5)
# plt.scatter(X_new_samples[:, 0], X_new_samples[:, 1], label='New minority samples', alpha=0.5)
plt.legend()
plt.show()
