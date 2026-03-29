import os
import joblib
import numpy as np
import pandas as pd

import sys
sys.path.append('..')
from config import MODEL_DIR


def load_model(model_name='best_model'):
    path = os.path.join(MODEL_DIR, f'{model_name}.joblib')
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model found at {path}. Run train.py first.")
    model = joblib.load(path)
    print(f"Loaded model from {path}")
    return model


def predict_transaction(transaction_dict, model=None):
    """
    Predict if a single transaction is fraudulent.

    transaction_dict should have keys: V1-V28, Amount_scaled, Time_scaled
    (already preprocessed features)
    """
    if model is None:
        model = load_model()

    # convert to dataframe with single row
    df = pd.DataFrame([transaction_dict])

    # make sure columns are in the right order
    expected_cols = [f'V{i}' for i in range(1, 29)] + ['Amount_scaled', 'Time_scaled']
    df = df[expected_cols]

    prediction = model.predict(df)[0]
    proba = model.predict_proba(df)[0]

    result = {
        'prediction': int(prediction),
        'label': 'FRAUD' if prediction == 1 else 'NOT FRAUD',
        'confidence': float(max(proba)),
        'fraud_probability': float(proba[1])
    }

    return result


# TODO: add batch prediction for CSV files
def predict_batch(csv_path, model=None):
    """predict on a whole CSV file - not fully implemented yet"""
    if model is None:
        model = load_model()
    df = pd.read_csv(csv_path)
    predictions = model.predict(df)
    return predictions


if __name__ == '__main__':
    # demo with a fake transaction (random values)
    np.random.seed(42)
    fake_transaction = {f'V{i}': np.random.randn() for i in range(1, 29)}
    fake_transaction['Amount_scaled'] = 0.5
    fake_transaction['Time_scaled'] = -0.3

    try:
        result = predict_transaction(fake_transaction)
        print(f"\nPrediction: {result['label']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Fraud probability: {result['fraud_probability']:.4f}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Train a model first by running: python -m src.train")
