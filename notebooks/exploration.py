"""
Quick EDA script for the credit card fraud dataset.
Run this first to understand the data before training.

Dataset: Kaggle Credit Card Fraud Detection
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_PATH


def class_distribution(df):
    print("\n=== Class Distribution ===")
    counts = df['Class'].value_counts()
    print(f"Non-Fraud (0): {counts[0]} ({counts[0]/len(df)*100:.2f}%)")
    print(f"Fraud (1):     {counts[1]} ({counts[1]/len(df)*100:.2f}%)")
    print(f"Ratio: 1:{counts[0]//counts[1]}")

    # plot it
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'])
    ax.set_xticklabels(['Non-Fraud', 'Fraud'], rotation=0)
    ax.set_title('Class Distribution')
    ax.set_ylabel('Count')
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=150)
    print("Saved class_distribution.png")
    plt.close()


def amount_distribution(df):
    print("\n=== Transaction Amount Stats ===")
    print(df['Amount'].describe())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # all transactions
    axes[0].hist(df[df['Class']==0]['Amount'], bins=50, alpha=0.7, label='Non-Fraud', color='#2ecc71')
    axes[0].hist(df[df['Class']==1]['Amount'], bins=50, alpha=0.7, label='Fraud', color='#e74c3c')
    axes[0].set_title('Transaction Amount Distribution')
    axes[0].set_xlabel('Amount')
    axes[0].legend()
    axes[0].set_xlim(0, 500)  # most transactions are under 500

    # log scale for better visibility
    axes[1].hist(df[df['Class']==0]['Amount'].apply(lambda x: np.log1p(x)),
                 bins=50, alpha=0.7, label='Non-Fraud', color='#2ecc71')
    axes[1].hist(df[df['Class']==1]['Amount'].apply(lambda x: np.log1p(x)),
                 bins=50, alpha=0.7, label='Fraud', color='#e74c3c')
    axes[1].set_title('Log Amount Distribution')
    axes[1].set_xlabel('log(Amount + 1)')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('amount_distribution.png', dpi=150)
    print("Saved amount_distribution.png")
    plt.close()


def correlation_heatmap(df):
    """correlation matrix - mostly just to see if V features are correlated with Class"""
    print("\n=== Top Correlations with Fraud ===")
    corr = df.corr()
    fraud_corr = corr['Class'].drop('Class').sort_values(key=abs, ascending=False)
    print(fraud_corr.head(10))

    # full heatmap is too big, just show top features
    top_features = fraud_corr.head(10).index.tolist() + ['Class']
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[top_features].corr(), annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, ax=ax)
    ax.set_title('Correlation Heatmap (Top Features)')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=150)
    print("Saved correlation_heatmap.png")
    plt.close()


def time_analysis(df):
    print("\n=== Time Analysis ===")
    # Time is seconds elapsed from first transaction
    df['Hour'] = df['Time'].apply(lambda x: (x // 3600) % 24)

    fig, ax = plt.subplots(figsize=(10, 5))
    fraud_by_hour = df[df['Class']==1].groupby('Hour').size()
    legit_by_hour = df[df['Class']==0].groupby('Hour').size()

    ax.bar(legit_by_hour.index, legit_by_hour.values, alpha=0.5, label='Non-Fraud', color='#2ecc71')
    ax.bar(fraud_by_hour.index, fraud_by_hour.values, alpha=0.7, label='Fraud', color='#e74c3c')
    ax.set_title("Transactions by Hour")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    plt.savefig('time_analysis.png', dpi=150)
    print("Saved time_analysis.png")
    plt.close()

    df.drop('Hour', axis=1, inplace=True)


if __name__ == '__main__':
    print("Loading dataset...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Dataset not found at {DATA_PATH}")
        print("Download it from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("Place creditcard.csv in the data/ folder")
        exit(1)

    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    class_distribution(df)
    amount_distribution(df)
    correlation_heatmap(df)
    time_analysis(df)

    print("\n--- EDA complete! Check the generated PNG files ---")
