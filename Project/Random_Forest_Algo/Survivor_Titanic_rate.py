import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('Titanic-Dataset.csv')
x = dataset.drop(['Survived', 'Name', 'PassengerId','Cabin'], axis=1)
y = dataset['Survived']
x['Age'] = x['Age'].fillna(x['Age'].mean())
x['Embarked'] = x['Embarked'].fillna('S')
x['Pclass'] = x['Pclass'].astype(str)
x['Ticket'] = x['Ticket'].astype(str)

column_to_one_hot = ['Sex', 'Pclass', 'Embarked','Ticket']
x = pd.get_dummies(x, columns=column_to_one_hot)

X_numpy = x.values
y_numpy = y.values

x_train, x_test, y_train, y_test = train_test_split(X_numpy, y_numpy, test_size=0.33, random_state=42)

class Node:
    def __init__(self,feature_index = None, left = None, right = None, value = None, threshold = None):
        self.feature_index = feature_index
        self.left = left
        self.right = right
        self.value = value
        self.threshold = threshold

class Decision_tree:
    def __init__(self, max_depth = 40, min_samples_split = 3, min_IG = 0.01):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_IG = min_IG

    def entropy(self, p):
        if p == 0 or p == 1 :
            return 0
        else :
            return -p*np.log2(p) - (1-p)*np.log2(1-p)

    def compute_IG(self, y, left_indices, right_indices):
        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0
        w1 = len(left_indices)/len(y)
        w2 = len(right_indices)/len(y)
        p1 = np.mean(y[left_indices])
        p2 = np.mean(y[right_indices])
        p_root = np.mean(y)
        IG = self.entropy(p_root) - (w1 * self.entropy(p1) + w2 * self.entropy(p2))
        return IG

    def split_indices(self, X, index,threshold):
        column_values = X[:, index]

        left_indices = np.where(column_values <= threshold)[0]
        right_indices = np.where(column_values > threshold)[0]

        return left_indices, right_indices

    def build_tree(self,X,y,curr_depth = 0):
        n_samples, n_features = X.shape
        n_label = len(np.unique(y))
        if curr_depth > self.max_depth or n_samples < self.min_samples_split or n_label == 1:
            return Node(value = round(np.mean(y)))

        best_IG = -1
        best_left = None
        best_right = None
        best_index = -1
        best_threshold = None

        for i in range(n_features):
            thresholds = np.unique(X[:, i])
            thresholds = np.random.choice(thresholds, int(np.sqrt(thresholds.shape[0])), replace=False)
            for threshold in thresholds:
                left_indices, right_indices = self.split_indices(X, i,threshold)
                if len(left_indices) > 0 and len(right_indices) > 0:
                    IG = self.compute_IG(y, left_indices, right_indices)
                    if IG > best_IG:
                        best_IG = IG
                        best_left = left_indices
                        best_right = right_indices
                        best_index = i
                        best_threshold = threshold

        if best_IG <= 0 :
            return Node(value = round(np.mean(y)))

        left_branch = self.build_tree(X[best_left], y[best_left], curr_depth + 1)
        right_branch = self.build_tree(X[best_right], y[best_right], curr_depth + 1)

        return Node(feature_index = best_index, left = left_branch, right = right_branch, threshold = best_threshold)

    def fit(self, X, y):
        self.root = self.build_tree(X, y,0)

    def make_predict(self, X,node):
        if node.value is not None: return node.value
        val = X[node.feature_index]

        if val <= node.threshold:
            return self.make_predict(X,node.left)
        else:
            return self.make_predict(X,node.right)

    def predict(self, X):
        y = np.array([self.make_predict(x,self.root) for x in X])
        return np.array(y)

class RandomForest:
    def __init__(self, max_depth = None, min_samples_split = None, min_IG = 0.01, n_estimators = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_estimators = n_estimators
        self.trees = []
        self.min_IG = min_IG

    def sample_replacement(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace = True)
        return X[idxs], y[idxs]

    def fit(self, X, y):
        self.trees = []
        for i in range(self.n_estimators):
            tree = Decision_tree(max_depth=self.max_depth, min_samples_split=self.min_samples_split,min_IG= self.min_IG)
            X1, y1 = self.sample_replacement(X, y)
            tree.fit(X1, y1)
            self.trees.append(tree)

    def predict(self,X):
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_predictions = np.swapaxes(tree_predictions, 0, 1)
        y_preds = []
        for prediction in tree_predictions :
            y_preds.append(round(np.mean(prediction)))
        return y_preds

for n in range (10,51,2):
    my_forest = RandomForest(min_samples_split=3,n_estimators=n, max_depth = 4)
    print("Đang huấn luyện Random Forest...")
    my_forest.fit(x_train, y_train)

    print("Đang dự đoán...")
    y_pred = my_forest.predict(x_test)

    correct_predictions = np.sum(y_pred == y_test)
    accuracy = correct_predictions / len(y_test)

    print("-" * 30)
    print(f"Số lượng cây: {my_forest.n_estimators}")
    print(f"Độ chính xác (Accuracy): {accuracy * 100:.2f}%")
    print("-" * 30)



