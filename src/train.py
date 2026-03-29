import os
import joblib
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import sys
sys.path.append('..')
from config import RANDOM_SEED, MODEL_DIR, MODEL_PARAMS
from src.preprocess import preprocess_pipeline


def train_logistic_regression(X_train, y_train):
    print("Training Logistic Regression...")
    params = MODEL_PARAMS['logistic_regression']
    lr = LogisticRegression(random_state=RANDOM_SEED, **params)

    start = time.time()
    lr.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s")

    return lr


def train_random_forest(X_train, y_train):
    print("Training Random Forest...")
    params = MODEL_PARAMS['random_forest']
    rf = RandomForestClassifier(random_state=RANDOM_SEED, **params)

    start = time.time()
    rf.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s")

    return rf


def train_xgboost(X_train, y_train):
    """XGBoost - works surprisingly well for imbalanced data even after SMOTE"""
    print("Training XGBoost...")
    params = MODEL_PARAMS['xgboost']
    xgb = XGBClassifier(random_state=RANDOM_SEED, **params)

    start = time.time()
    xgb.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s")

    return xgb


def save_model(model, name):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f'{name}.joblib')
    joblib.dump(model, path)
    print(f"Saved {name} to {path}")


def train_all_models(X_train, y_train):
    models = {}

    models['logistic_regression'] = train_logistic_regression(X_train, y_train)
    models['random_forest'] = train_random_forest(X_train, y_train)
    models['xgboost'] = train_xgboost(X_train, y_train)

    return models


# TODO: add cross-validation later
def find_best_model(models, X_test, y_test):
    """Compare models on test set and return the best one based on F1"""
    from sklearn.metrics import f1_score

    best_name = None
    best_f1 = 0

    print("\n--- Model Comparison ---")
    for name, model in models.items():
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        print(f"{name}: F1 = {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_name = name

    print(f"\nBest model: {best_name} (F1={best_f1:.4f})")
    return best_name, models[best_name]


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, scaler = preprocess_pipeline()

    print("\n=== Training Models ===\n")
    models = train_all_models(X_train, y_train)

    best_name, best_model = find_best_model(models, X_test, y_test)

    # save all models
    for name, model in models.items():
        save_model(model, name)

    # also save the best one with a clear name
    save_model(best_model, 'best_model')
    print("\nTraining pipeline complete!")
