import os
import argparse
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, accuracy_score, f1_score

try:
    import mlflow
    import mlflow.sklearn
    import mlflow.pytorch
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

from src.config import (
    DEVICE,
    MODELS_DIR,
    STAGE1_WEIGHTS_PATH,
    STAGE2_WEIGHTS_PATH,
    FEATURE_COLS,
    DISEASE_CLASSES,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
)
from src.models import AttentionUNet


def setup_mlflow():
    """Initialize MLflow tracking URI and experiment name if MLflow is available."""
    if HAS_MLFLOW:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def calculate_dice_score(pred, target, num_classes=4, smooth=1e-5):
    """Compute per-class and mean Dice Similarity Coefficients."""
    dice_scores = {}
    for cls in range(1, num_classes):
        p = (pred == cls).float()
        t = (target == cls).float()
        intersection = (p * t).sum()
        total = p.sum() + t.sum()
        dice = (2.0 * intersection + smooth) / (total + smooth)
        dice_scores[cls] = dice.item()
    mean_dice = sum(dice_scores.values()) / len(dice_scores)
    return mean_dice, dice_scores


def train_stage1_attention_unet(
    train_loader=None,
    val_loader=None,
    epochs=50,
    lr=1e-4,
    batch_size=8,
    save_path=STAGE1_WEIGHTS_PATH
):
    """
    Train Attention U-Net (Stage 1) with MLflow metrics and artifact logging.
    """
    setup_mlflow()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    model = AttentionUNet(n_channels=1, n_classes=4, bilinear=False).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    run_context = (
        mlflow.start_run(run_name="stage1_attention_unet_training")
        if HAS_MLFLOW else None
    )

    try:
        if HAS_MLFLOW:
            mlflow.log_params({
                "stage": "1_segmentation",
                "model_architecture": "AttentionUNet",
                "optimizer": "Adam",
                "learning_rate": lr,
                "batch_size": batch_size,
                "epochs": epochs,
                "loss_function": "CrossEntropyLoss",
                "device": str(DEVICE),
                "num_classes": 4
            })

        best_val_dice = 0.0

        if train_loader is not None and val_loader is not None:
            for epoch in range(1, epochs + 1):
                model.train()
                running_loss = 0.0

                for images, masks in train_loader:
                    images, masks = images.to(DEVICE), masks.to(DEVICE)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, masks.long())
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                train_loss = running_loss / max(1, len(train_loader))

                # Validation step
                model.eval()
                val_loss = 0.0
                all_dice = []
                with torch.no_grad():
                    for val_images, val_masks in val_loader:
                        val_images, val_masks = val_images.to(DEVICE), val_masks.to(DEVICE)
                        val_outputs = model(val_images)
                        v_loss = criterion(val_outputs, val_masks.long())
                        val_loss += v_loss.item()

                        preds = torch.argmax(val_outputs, dim=1)
                        m_dice, _ = calculate_dice_score(preds, val_masks)
                        all_dice.append(m_dice)

                val_loss = val_loss / max(1, len(val_loader))
                mean_dice = sum(all_dice) / max(1, len(all_dice))

                if HAS_MLFLOW:
                    mlflow.log_metrics({
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_mean_dice": mean_dice
                    }, step=epoch)

                if mean_dice > best_val_dice:
                    best_val_dice = mean_dice
                    torch.save(model.state_dict(), save_path)

        if HAS_MLFLOW:
            mlflow.log_metric("best_val_dice", best_val_dice)
            if os.path.exists(save_path):
                mlflow.log_artifact(save_path, artifact_path="stage1_model")

    finally:
        if run_context is not None:
            mlflow.end_run()

    return model


def train_stage2_disease_classifier(
    x_features,
    y_labels,
    n_estimators=150,
    max_depth=8,
    save_path=STAGE2_WEIGHTS_PATH
):
    """
    Train Random Forest Disease Classifier (Stage 2) with MLflow metrics and artifact logging.
    """
    setup_mlflow()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        class_weight="balanced"
    )

    run_context = (
        mlflow.start_run(run_name="stage2_random_forest_classification")
        if HAS_MLFLOW else None
    )

    try:
        if HAS_MLFLOW:
            mlflow.log_params({
                "stage": "2_pathology_classification",
                "model_type": "RandomForestClassifier",
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "random_state": 42,
                "num_features": len(FEATURE_COLS),
                "target_classes": DISEASE_CLASSES
            })

        # 5-Fold Stratified Cross-Validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = cross_validate(
            clf,
            x_features,
            y_labels,
            cv=cv,
            scoring=["accuracy", "f1_macro", "precision_macro", "recall_macro"]
        )

        mean_acc = float(np.mean(cv_results["test_accuracy"]))
        mean_f1 = float(np.mean(cv_results["test_f1_macro"]))
        mean_prec = float(np.mean(cv_results["test_precision_macro"]))
        mean_rec = float(np.mean(cv_results["test_recall_macro"]))

        # Fit full model
        clf.fit(x_features, y_labels)
        joblib.dump(clf, save_path)

        if HAS_MLFLOW:
            mlflow.log_metrics({
                "cv_mean_accuracy": mean_acc,
                "cv_mean_f1_macro": mean_f1,
                "cv_mean_precision_macro": mean_prec,
                "cv_mean_recall_macro": mean_rec
            })
            if os.path.exists(save_path):
                mlflow.log_artifact(save_path, artifact_path="stage2_model")

    finally:
        if run_context is not None:
            mlflow.end_run()

    return clf, {
        "mean_accuracy": mean_acc,
        "mean_f1": mean_f1
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Cardiac MRI Pipeline with MLflow Tracking")
    parser.add_argument("--stage", type=str, default="2", choices=["1", "2", "all"], help="Stage to train")
    args = parser.parse_args()

    print(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"MLflow Experiment: {MLFLOW_EXPERIMENT_NAME}")
