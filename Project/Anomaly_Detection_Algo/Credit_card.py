import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer

data_resource = pd.read_csv('creditcard.csv')
data_np = data_resource.values
pearson = np.zeros(data_np.shape[1] - 1)
avg_y = np.mean(data_np[:,30])
for i in range(pearson.shape[0]):
    avg_x = np.mean(data_np[:,i])
    pearson[i] =(np.sum((data_np[:,i] - avg_x) * (data_np[:,30] - avg_y)) / (np.sqrt(np.sum((data_np[:,i] - avg_x) ** 2)) * np.sqrt(np.sum((data_np[:,30] - avg_y) ** 2))))
features = []
for feature_idx in range(pearson.shape[0]):
    if pearson[feature_idx] >= 0.22 or pearson[feature_idx] <= -0.22:
        features.append(feature_idx)
features.append(30)
features = np.array(features)
new_data = data_resource.iloc[:,features]

y_0 = new_data[new_data['Class'] == 0]
y_1 = new_data[new_data['Class'] == 1]
print(y_0.shape,y_1.shape)
X_train, X_cv = train_test_split(y_0, test_size=0.34,random_state= 42)
y1_cv = y_1
X_cv = pd.concat([X_cv,y1_cv])

def compute_mean_var(X):
    mean = np.mean(X, axis=0)
    var = np.var(X, axis=0)
    return mean, var

def compute_p_gaussian(X, mean, var):
    p = np.ones(X.shape[0])
    mean = mean.values
    var = var.values
    X = X.values
    for i in range(X.shape[1] - 1):
        p = p * (1 / np.sqrt(2 * np.pi * var[i]) * np.exp(-((X[:, i] - mean[i]) ** 2) / (2 * var[i])))
    return p

def find_best_epsilon(X, p_value):
    best_F1 = 0
    best_epsilon = 0
    best_recall = 0
    best_precision = 0
    for epsilon in p_value:
        preicision = (p_value < epsilon)
        tp = np.sum((preicision == 1) & (X['Class'] == 1))
        fp = np.sum((preicision == 1) & (X['Class'] == 0))
        fn = np.sum((preicision == 0) & (X['Class'] == 1))
        recall = 0
        precision = 0
        if tp + fn  > 0:
            recall = tp / (tp + fn)
        if tp + fp > 0:
            precision = tp / (tp + fp)

        if recall + precision > 0:
            F1 = 2 * precision * recall / (precision + recall)
        else:
            F1 = 0

        if F1 > best_F1:
            best_epsilon = epsilon
            best_F1 = F1
            best_recall = recall
            best_precision = precision

    return best_epsilon, best_F1,best_recall, best_precision


mean, var = compute_mean_var(X_train)
p_cv = compute_p_gaussian(X_cv, mean, var)
epsilon, F1,recall,precision = find_best_epsilon(X_cv, p_cv)
print('complited!!!!',epsilon)
print("recall: ",recall*100)
print("precision: ",precision*100)
print("F1: ",F1)




