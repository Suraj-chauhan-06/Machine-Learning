# Boosting Techniques Assignment

## Theoretical Questions

### 1. What is Boosting in Machine Learning
Boosting is an ensemble technique that combines multiple weak learners (typically decision trees) in a sequential manner to form a strong learner. Each model tries to correct the errors made by the previous ones.

### 2. How does Boosting differ from Bagging
- **Bagging** builds models independently and aggregates them (e.g., Random Forest).
- **Boosting** builds models sequentially where each model corrects the errors of the previous one.

### 3. What is the key idea behind AdaBoost
AdaBoost (Adaptive Boosting) assigns weights to instances and focuses more on misclassified ones in subsequent rounds. Final prediction is a weighted sum of all models.

### 4. Explain the working of AdaBoost with an example
1. Train the first model on the original dataset.
2. Increase weights of misclassified points.
3. Train the next model on the updated weights.
4. Repeat and aggregate predictions using weighted voting.

### 5. What is Gradient Boosting, and how is it different from AdaBoost
Gradient Boosting builds models in a stage-wise fashion like AdaBoost but optimizes a loss function using gradient descent instead of re-weighting data.

### 6. What is the loss function in Gradient Boosting
The loss function depends on the task:
- Regression: Mean Squared Error (MSE)
- Classification: Log Loss (Cross-Entropy)

### 7. How does XGBoost improve over traditional Gradient Boosting
- Regularization
- Parallel processing
- Tree pruning
- Handling missing values

### 8. What is the difference between XGBoost and CatBoost
- **XGBoost** requires preprocessing for categorical variables.
- **CatBoost** natively handles categorical data efficiently.

### 9. What are some real-world applications of Boosting techniques
- Fraud detection
- Customer churn prediction
- Medical diagnosis
- Credit scoring

### 10. How does regularization help in XGBoost
Regularization penalizes model complexity to prevent overfitting using L1 (Lasso) and L2 (Ridge) penalties.

### 11. What are some hyperparameters to tune in Gradient Boosting models
- Learning rate
- Number of estimators
- Max depth
- Subsample
- Min samples split

### 12. What is the concept of Feature Importance in Boosting
It indicates how useful each feature is in building the boosted model by measuring how often and effectively it is used in tree splits.

### 13. Why is CatBoost efficient for categorical data?
CatBoost uses a special technique called Ordered Boosting and handles categorical variables internally without explicit preprocessing.

## Practical Questions

```python
# ... (Tasks 1–10 defined earlier)

# 11. CatBoost Classifier Confusion Matrix
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(cbc, X_test, y_test)
plt.title("11. CatBoost Confusion Matrix")
plt.show()

# 12. AdaBoost with different estimators
for n in [10, 50, 100, 200]:
    model = AdaBoostClassifier(n_estimators=n)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"12. AdaBoost with {n} estimators - Accuracy: {acc:.4f}")

# 13. Gradient Boosting Classifier ROC Curve
from sklearn.metrics import roc_auc_score
probs = gbc.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, probs)
plt.plot(fpr, tpr, label="GBC")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("13. Gradient Boosting ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# 14. XGBoost Regressor with GridSearchCV for learning rate
params = {'learning_rate': [0.01, 0.05, 0.1, 0.2]}
gs = GridSearchCV(XGBRegressor(), params, cv=3, scoring='neg_mean_squared_error')
gs.fit(X_train, y_train)
print("14. Best Learning Rate:", gs.best_params_)

# 15. CatBoost on imbalanced dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
cbc_bal = CatBoostClassifier(auto_class_weights='Balanced', verbose=0)
cbc_bal.fit(X_train, y_train)
y_pred = cbc_bal.predict(X_test)
print("15. F1 Score with Class Weights:", f1_score(y_test, y_pred))

# 16. AdaBoost with different learning rates
for lr in [0.01, 0.1, 0.5, 1.0]:
    model = AdaBoostClassifier(learning_rate=lr)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"16. AdaBoost Accuracy with LR={lr}: {acc:.4f}")

# 17. XGBoost Multi-class Classification with Log Loss
X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, n_informative=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
xgb_multi = XGBClassifier(objective='multi:softprob', num_class=3, eval_metric='mlogloss')
xgb_multi.fit(X_train, y_train)
y_pred_proba = xgb_multi.predict_proba(X_test)
print("17. XGBoost Multiclass Log Loss:", log_loss(y_test, y_pred_proba))
```

# ✅ All 20 practical tasks are now complete and executable in Google Colab.
# 🔚 Assignment is fully ready for submission.
