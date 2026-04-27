import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import uniform, randint
from collections import Counter
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, classification_report


# 1. Load dataset
file_path = r'PATH'
df = pd.read_excel(file_path)

# 2. Define features and target
X = df.drop(columns=['FEATURE'])
y = df['FEATURE']

# 3. Train-test split for internal validation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_final = X_train
y_train_final = y_train

print(f"Training class distribution: {Counter(y_train_final)}")

# 4. Initialize classifier
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    missing=float('nan')
)

# 5. Hyperparameter search space
param_dist = {
    'learning_rate': uniform(0.01, 0.15),
    'max_depth': randint(3, 7),
    'colsample_bytree': uniform(0.4, 0.6),
    'colsample_bylevel': uniform(0.4, 0.6),
    'subsample': uniform(0.3, 0.6),
    'n_estimators': randint(200, 500),
    'alpha': uniform(0, 5.0),
    'reg_lambda': uniform(0.5, 9.5),
    'gamma': uniform(0, 5.0),
    'min_child_weight': randint(1, 10)
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=50,
    scoring='roc_auc',
    cv=10,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

print("Running 10-fold cross-validation...")
random_search.fit(X_train_final, y_train_final)

# 7. Extract best model and evaluate
best_xgb_model = random_search.best_estimator_
print(f"\nBest hyperparameters: {random_search.best_params_}")
print(f"Best CV AUROC: {random_search.best_score_:.4f}")

y_pred_proba = best_xgb_model.predict_proba(X_test)[:, 1]
test_auroc = roc_auc_score(y_test, y_pred_proba)
print(f"Internal hold-out AUROC: {test_auroc:.4f}")




