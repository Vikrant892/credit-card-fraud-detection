import os

# paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'creditcard.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

RANDOM_SEED = 42
TEST_SIZE = 0.2

# model hyperparams - tuned these manually, grid search was taking too long
MODEL_PARAMS = {
    'logistic_regression': {
        'max_iter': 1000,
        'C': 0.1,
        'solver': 'lbfgs'
    },
    'random_forest': {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 5,
        'n_jobs': -1
    },
    'xgboost': {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'logloss',
        'use_label_encoder': False
    }
}

# SMOTE oversampling ratio
SMOTE_RATIO = 0.5  # don't go 1:1, found that 0.5 works better
