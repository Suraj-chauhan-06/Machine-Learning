# Decision Tree Assignment - Google Colab Format

## Part 1: Theoretical Questions

### 1. What is a Decision Tree, and how does it work?
A Decision Tree is a supervised learning algorithm used for classification and regression tasks. It splits the data into subsets based on the value of input features, building a tree where each internal node represents a feature test, each branch a test outcome, and each leaf a class label or value.

### 2. What are impurity measures in Decision Trees?
Impurity measures quantify the homogeneity of target labels in a node. Common impurity measures include Gini Impurity and Entropy.

### 3. What is the mathematical formula for Gini Impurity?
Gini(D) = 1 - \sum(p_i^2), where p_i is the probability of class i in dataset D.

### 4. What is the mathematical formula for Entropy?
Entropy(D) = - \sum(p_i * log2(p_i)), where p_i is the probability of class i in dataset D.

### 5. What is Information Gain, and how is it used in Decision Trees?
Information Gain measures the reduction in entropy after a dataset is split on an attribute. It's used to decide the best feature to split on at each step in building the tree.

### 6. What is the difference between Gini Impurity and Entropy?
Both measure node impurity. Gini is faster to compute, while Entropy can be more informative for skewed data.

### 7. What is the mathematical explanation behind Decision Trees?
Decision Trees use a greedy algorithm to minimize impurity at each node by selecting the best split based on Information Gain or Gini Impurity.

### 8. What is Pre-Pruning in Decision Trees?
Pre-pruning stops tree growth early by imposing constraints like max_depth or min_samples_split.

### 9. What is Post-Pruning in Decision Trees?
Post-pruning grows the full tree and then removes branches that do not provide significant gain, typically using validation data.

### 10. What is the difference between Pre-Pruning and Post-Pruning?
Pre-pruning limits tree growth during training, while post-pruning simplifies a full tree after training.

### 11. What is a Decision Tree Regressor?
A Decision Tree Regressor predicts continuous values instead of categories, minimizing variance within splits.

### 12. What are the advantages and disadvantages of Decision Trees?
Advantages:
- Simple and interpretable
- Requires little data preprocessing
Disadvantages:
- Prone to overfitting
- Unstable to small data changes

### 13. How does a Decision Tree handle missing values?
It can either ignore missing values, impute them, or use surrogate splits.

### 14. How does a Decision Tree handle categorical features?
By splitting nodes based on each category or by encoding categories numerically.

### 15. What are some real-world applications of Decision Trees?
- Medical diagnosis
- Customer churn prediction
- Loan approval
- Fraud detection

---

## Part 2: Python Code (Practical Exercises)

```python
# Common imports
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Train on Iris dataset and print accuracy
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

# 2. Train using Gini Impurity and print feature importances
model_gini = DecisionTreeClassifier(criterion='gini')
model_gini.fit(X_train, y_train)
print("Feature importances:", model_gini.feature_importances_)

# 3. Train using Entropy and print accuracy
model_entropy = DecisionTreeClassifier(criterion='entropy')
model_entropy.fit(X_train, y_train)
print("Accuracy (Entropy):", accuracy_score(y_test, model_entropy.predict(X_test)))

# 4. Decision Tree Regressor on housing dataset
housing = fetch_california_housing()
Xh_train, Xh_test, yh_train, yh_test = train_test_split(housing.data, housing.target, random_state=42)
regressor = DecisionTreeRegressor()
regressor.fit(Xh_train, yh_train)
print("MSE:", mean_squared_error(yh_test, regressor.predict(Xh_test)))

# 5. Visualize tree using plot_tree
plt.figure(figsize=(12, 8))
plot_tree(model, filled=True, feature_names=iris.feature_names)
plt.show()

# 6. Train with max_depth=3
model_depth3 = DecisionTreeClassifier(max_depth=3)
model_depth3.fit(X_train, y_train)
print("Accuracy (Depth=3):", accuracy_score(y_test, model_depth3.predict(X_test)))

# 7. Train with min_samples_split=5
model_split5 = DecisionTreeClassifier(min_samples_split=5)
model_split5.fit(X_train, y_train)
print("Accuracy (min_samples_split=5):", accuracy_score(y_test, model_split5.predict(X_test)))

# 8. Apply feature scaling (not required for trees but shown for comparison)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(iris.data)
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_scaled, iris.target, random_state=42)
model_scaled = DecisionTreeClassifier()
model_scaled.fit(X_train_s, y_train_s)
print("Accuracy with scaled data:", accuracy_score(y_test_s, model_scaled.predict(X_test_s)))

# 9. One-vs-Rest strategy (default in sklearn)
print("OvR Accuracy:", accuracy_score(y_test, model.predict(X_test)))

# 10. Display feature importance scores
print("Feature Importances:", model.feature_importances_)

# 11. Regressor with max_depth=5
reg5 = DecisionTreeRegressor(max_depth=5)
reg5.fit(Xh_train, yh_train)
print("MSE (Depth=5):", mean_squared_error(yh_test, reg5.predict(Xh_test)))

# 12. Cost Complexity Pruning (CCP)
path = model.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
accuracies = []
for alpha in ccp_alphas:
    clf = DecisionTreeClassifier(ccp_alpha=alpha)
    clf.fit(X_train, y_train)
    accuracies.append(accuracy_score(y_test, clf.predict(X_test)))
plt.plot(ccp_alphas, accuracies)
plt.xlabel("Alpha")
plt.ylabel("Accuracy")
plt.title("Cost Complexity Pruning")
plt.show()

# 13. Precision, Recall, F1
y_pred = model.predict(X_test)
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall:", recall_score(y_test, y_pred, average='macro'))
print("F1-Score:", f1_score(y_test, y_pred, average='macro'))

# 14. Confusion Matrix with seaborn
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 15. GridSearchCV for hyperparameter tuning
params = {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5, 10]}
gs = GridSearchCV(DecisionTreeClassifier(), params, cv=5)
gs.fit(X_train, y_train)
print("Best parameters:", gs.best_params_)
print("Best score:", gs.best_score_)
```

---
You can copy this notebook content directly into Google Colab and run each cell to complete your assignment.
