import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend, this took forever to figure out
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, f1_score
)


def print_metrics(y_true, y_pred, model_name="Model"):
    print(f"\n{'='*50}")
    print(f"  {model_name} - Classification Report")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred, target_names=['Non-Fraud', 'Fraud']))


def plot_confusion_matrix(y_true, y_pred, model_name="Model", save_path=None):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Fraud', 'Fraud'],
                yticklabels=['Non-Fraud', 'Fraud'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved confusion matrix to {save_path}")
    plt.close()


def plot_roc_curve(y_true, y_prob, model_name="Model", save_path=None):
    """Plot ROC curve. y_prob should be probability of positive class."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved ROC curve to {save_path}")
    plt.close()

    return roc_auc


def plot_precision_recall(y_true, y_prob, model_name="Model", save_path=None):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved PR curve to {save_path}")
    plt.close()


def evaluate_model(model, X_test, y_test, model_name="Model", output_dir=None):
    y_pred = model.predict(X_test)

    # some models don't have predict_proba (but all of ours do)
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred  # fallback

    print_metrics(y_test, y_pred, model_name)

    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)

        cm_path = os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
        roc_path = os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_roc_curve.png')
        pr_path = os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_pr_curve.png')

        plot_confusion_matrix(y_test, y_pred, model_name, cm_path)
        roc_auc = plot_roc_curve(y_test, y_prob, model_name, roc_path)
        plot_precision_recall(y_test, y_prob, model_name, pr_path)

        print(f"AUC-ROC: {roc_auc:.4f}")

    f1 = f1_score(y_test, y_pred)
    return f1


if __name__ == '__main__':
    import joblib
    import os
    import sys
    sys.path.append('..')
    from config import MODEL_DIR
    from src.preprocess import preprocess_pipeline

    X_train, X_test, y_train, y_test, scaler = preprocess_pipeline()

    # evaluate all saved models
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.joblib') and f != 'best_model.joblib']

    for mf in model_files:
        model = joblib.load(os.path.join(MODEL_DIR, mf))
        name = mf.replace('.joblib', '').replace('_', ' ').title()
        evaluate_model(model, X_test, y_test, name, output_dir='outputs')
