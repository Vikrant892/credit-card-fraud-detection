# Credit Card Fraud Detection

Built this during my Masters coursework at the University of Adelaide. The goal was to build an end-to-end ML pipeline that detects fraudulent credit card transactions using the popular Kaggle dataset.

The dataset is massively imbalanced (only 0.17% of transactions are fraud), so the main challenge was figuring out how to handle that. I ended up using SMOTE oversampling which made a huge difference — without it, the models basically just predicted everything as non-fraud and still got 99%+ accuracy, which is obviously useless.

I trained three models — Logistic Regression, Random Forest, and XGBoost — and compared them. XGBoost performed the best overall, though Random Forest was close.

## Dataset

Using the [Credit Card Fraud Detection dataset from Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). It has 284,807 transactions with 30 features (V1-V28 are PCA-transformed, plus Amount and Time). Download it and put `creditcard.csv` in the `data/` folder.

## Setup

```bash
pip install -r requirements.txt
```

Download the dataset and place it in `data/creditcard.csv`.

## How to Run

```bash
# EDA first
python notebooks/exploration.py

# Train all models
python -m src.train

# Evaluate (after training)
python -m src.evaluate

# Predict on a single transaction
python -m src.predict
```

## Results

Best model: **XGBoost** with SMOTE oversampling

| Metric | Logistic Regression | Random Forest | XGBoost |
|--------|-------------------|---------------|---------|
| Precision | 0.88 | 0.93 | 0.91 |
| Recall | 0.62 | 0.80 | 0.84 |
| F1 Score | 0.73 | 0.86 | 0.87 |
| AUC-ROC | 0.97 | 0.98 | 0.98 |

XGBoost had the best balance of precision and recall. Random Forest had slightly higher precision but lower recall.

## Project Structure

```
├── config.py          # paths, hyperparameters
├── src/
│   ├── preprocess.py  # data loading, scaling, SMOTE
│   ├── train.py       # model training pipeline
│   ├── evaluate.py    # metrics, plots
│   └── predict.py     # single transaction prediction
├── notebooks/
│   └── exploration.py # EDA script
├── data/              # put creditcard.csv here
└── models/            # saved models (generated)
```

## What I Learned

- Class imbalance is a real pain. Accuracy is a terrible metric when 99.8% of your data is one class. F1 score and AUC-ROC are much more meaningful.
- SMOTE helps a lot but you have to be careful with the sampling ratio. Going full 1:1 actually made things worse — 0.5 was the sweet spot.
- XGBoost is impressively good out of the box. Even without much hyperparameter tuning it beat the other models.
- Feature importance from Random Forest showed V14, V4, V12 were the most predictive features, which makes sense since those had the highest correlation with the target.

## Future Work

- Add cross-validation for more robust evaluation
- Try neural network approach (autoencoder for anomaly detection)
- Deploy as a REST API with Flask
- Experiment with different SMOTE variants (ADASYN, BorderlineSMOTE)
