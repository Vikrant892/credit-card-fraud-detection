"""
Data preprocessing pipeline for credit card fraud detection.
Handles loading, cleaning, scaling, and oversampling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

import sys
sys.path.append('..')
from config import DATA_PATH, RANDOM_SEED, TEST_SIZE, SMOTE_RATIO


def load_data(path=None):
    """Load the creditcard.csv dataset from Kaggle.
    Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    """
    if path is None:
        path = DATA_PATH

    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} transactions")
    print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.2f}%)")

    return df


def handle_missing(df):
    # the kaggle dataset is pretty clean but just in case
    if df.isnull().sum().sum() > 0:
        print(f"Found {df.isnull().sum().sum()} missing values, filling with median")
        df = df.fillna(df.median())
    else:
        print("No missing values found")
    return df


def scale_features(df):
    """Scale Amount and Time columns - the V1-V28 features are already PCA transformed
    so they don't need scaling"""
    scaler = StandardScaler()

    df = df.copy()
    df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
    df['Time_scaled'] = scaler.fit_transform(df[['Time']])

    # drop original columns
    df.drop(['Amount', 'Time'], axis=1, inplace=True)

    return df, scaler


def split_data(df):
    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Train fraud ratio: {y_train.mean()*100:.2f}%")

    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train):
    """Apply SMOTE oversampling to balance the training data.
    This was the key insight - without SMOTE the models just predict everything as non-fraud"""
    print("\nApplying SMOTE oversampling...")
    print(f"Before SMOTE - Fraud: {y_train.sum()}, Non-fraud: {len(y_train) - y_train.sum()}")

    smote = SMOTE(sampling_strategy=SMOTE_RATIO, random_state=RANDOM_SEED)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print(f"After SMOTE - Fraud: {y_resampled.sum()}, Non-fraud: {len(y_resampled) - y_resampled.sum()}")

    return X_resampled, y_resampled


def preprocess_pipeline(path=None):
    """Run the full preprocessing pipeline and return train/test splits."""
    df = load_data(path)
    df = handle_missing(df)
    df, scaler = scale_features(df)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    print("\n--- Preprocessing complete ---")
    return X_train_res, X_test, y_train_res, y_test, scaler


if __name__ == '__main__':
    # quick test
    X_train, X_test, y_train, y_test, scaler = preprocess_pipeline()
    print(f"\nFinal shapes: X_train={X_train.shape}, X_test={X_test.shape}")
