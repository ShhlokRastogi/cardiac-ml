import os
import json
import numpy as np
import scipy.ndimage as ndimage
from sklearn.metrics import classification_report, confusion_matrix, r2_score

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

from src.config import (
    DISEASE_CLASSES,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
)


def setup_mlflow():
    """Initialize MLflow tracking URI and experiment name if available."""
    if HAS_MLFLOW:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def compute_dice_score(pred_mask, gt_mask, class_idx):
    """Compute Dice Similarity Coefficient for a specific anatomical class."""
    pred_bin = (pred_mask == class_idx).astype(np.float32)
    gt_bin = (gt_mask == class_idx).astype(np.float32)
    intersection = np.sum(pred_bin * gt_bin)
    total = np.sum(pred_bin) + np.sum(gt_bin)
    if total == 0:
        return 1.0 if np.sum(gt_bin) == 0 else 0.0
    return float((2.0 * intersection) / total)


def evaluate_segmentation_predictions(all_preds, all_gts):
    """
    Evaluate 3D multi-class segmentation predictions across dataset.
    Returns Dice metrics for RV (1), Myocardium (2), and LV (3).
    """
    rv_dices, myo_dices, lv_dices = [], [], []

    for pred, gt in zip(all_preds, all_gts):
        rv_dices.append(compute_dice_score(pred, gt, 1))
        myo_dices.append(compute_dice_score(pred, gt, 2))
        lv_dices.append(compute_dice_score(pred, gt, 3))

    results = {
        "mean_dice_rv": float(np.mean(rv_dices)),
        "mean_dice_myo": float(np.mean(myo_dices)),
        "mean_dice_lv": float(np.mean(lv_dices)),
        "mean_dice_overall": float(np.mean([np.mean(rv_dices), np.mean(myo_dices), np.mean(lv_dices)]))
    }
    return results


def log_evaluation_to_mlflow(eval_metrics, run_name="pipeline_evaluation", artifacts_dict=None):
    """
    Log evaluation metrics and diagnostic reports to MLflow.
    """
    setup_mlflow()
    if not HAS_MLFLOW:
        return

    with mlflow.start_run(run_name=run_name):
        mlflow.log_metrics(eval_metrics)
        if artifacts_dict:
            for art_name, art_content in artifacts_dict.items():
                temp_path = f"{art_name}.json"
                with open(temp_path, "w") as f:
                    json.dump(art_content, f, indent=2)
                mlflow.log_artifact(temp_path, artifact_path="evaluation_reports")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
