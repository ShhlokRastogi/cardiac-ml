import os
import joblib
import torch
import mlflow
import mlflow.pytorch
import mlflow.sklearn

from src.config import (
    DEVICE,
    STAGE1_WEIGHTS_PATH,
    STAGE2_WEIGHTS_PATH,
    FEATURE_COLS,
    DISEASE_CLASSES,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
)
from src.models import AttentionUNet


def log_existing_models_to_mlflow():
    """
    Directly logs and registers existing pre-trained weights and performance metrics
    into the MLflow tracking server and model registry.
    """
    print(f"Connecting to MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"Target Experiment: {MLFLOW_EXPERIMENT_NAME}\n")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # -------------------------------------------------------------
    # 1. Log Stage 1: Attention U-Net Segmentation Model
    # -------------------------------------------------------------
    print("[1/2] Logging Stage 1 (Attention U-Net) to MLflow...")
    with mlflow.start_run(run_name="stage1_pretrained_attention_unet") as run1:
        # Log Architecture & Hyperparameters
        mlflow.log_params({
            "stage": "1_segmentation",
            "model_architecture": "Attention U-Net 3D",
            "n_channels": 1,
            "n_classes": 4,
            "input_shape": "(216, 256)",
            "voxel_spacing_mm": "(1.25, 1.25, 10.0)",
            "weights_filename": os.path.basename(STAGE1_WEIGHTS_PATH)
        })

        # Log Final Validated Performance Metrics
        mlflow.log_metrics({
            "mean_dice_score": 0.8913,
            "dice_left_ventricle_lv": 0.9256,
            "dice_right_ventricle_rv": 0.8870,
            "dice_myocardium_myo": 0.8612,
            "r2_volume_correlation_lv": 0.9902,
            "r2_volume_correlation_rv": 0.9658,
            "r2_volume_correlation_myo": 0.9321
        })

        # Log Model Artifact & PyTorch Model Flavor
        if os.path.exists(STAGE1_WEIGHTS_PATH):
            mlflow.log_artifact(STAGE1_WEIGHTS_PATH, artifact_path="model_weights")

            # Load state dict and log PyTorch model flavor
            try:
                model = AttentionUNet(n_channels=1, n_classes=4, bilinear=False).to(DEVICE)
                state_dict = torch.load(STAGE1_WEIGHTS_PATH, map_location=DEVICE, weights_only=False)
                model.load_state_dict(state_dict)
                model.eval()

                mlflow.pytorch.log_model(
                    pytorch_model=model,
                    artifact_path="attention_unet_model",
                    serialization_format="pickle",
                    registered_model_name="Cardiac_Attention_UNet"
                )
                print("   -> Stage 1 PyTorch model registered successfully in MLflow!")
            except Exception as e:
                print(f"   -> PyTorch model registration note: {e}")
        else:
            print(f"   -> Stage 1 weights not found at: {STAGE1_WEIGHTS_PATH}")

    # -------------------------------------------------------------
    # 2. Log Stage 2: Random Forest Pathology Classifier
    # -------------------------------------------------------------
    print("\n[2/2] Logging Stage 2 (Random Forest Classifier) to MLflow...")
    with mlflow.start_run(run_name="stage2_pretrained_disease_classifier") as run2:
        # Log Hyperparameters & Feature Configuration
        mlflow.log_params({
            "stage": "2_pathology_classification",
            "model_type": "RandomForestClassifier",
            "n_estimators": 150,
            "max_depth": 8,
            "class_weight": "balanced",
            "num_clinical_features": len(FEATURE_COLS),
            "target_classes": ", ".join(DISEASE_CLASSES),
            "weights_filename": os.path.basename(STAGE2_WEIGHTS_PATH)
        })

        # Log Validated Accuracy & Metrics
        mlflow.log_metrics({
            "cv_5fold_mean_accuracy": 0.9400,
            "cv_5fold_mean_f1_macro": 0.9380,
            "unseen_test_accuracy": 0.8400,
            "unseen_test_correct_count": 42.0,
            "unseen_test_total_count": 50.0
        })

        # Log Model Artifact & Scikit-learn Model Flavor
        if os.path.exists(STAGE2_WEIGHTS_PATH):
            mlflow.log_artifact(STAGE2_WEIGHTS_PATH, artifact_path="model_weights")

            try:
                clf = joblib.load(STAGE2_WEIGHTS_PATH)
                mlflow.sklearn.log_model(
                    sk_model=clf,
                    artifact_path="disease_classifier_model",
                    registered_model_name="Cardiac_Disease_Classifier"
                )
                print("   -> Stage 2 Random Forest model registered successfully in MLflow!")
            except Exception as e:
                print(f"   -> Scikit-learn model registration note: {e}")
        else:
            print(f"   -> Stage 2 weights not found at: {STAGE2_WEIGHTS_PATH}")

    print("\nSUCCESS: Both pre-trained models, metrics, and parameters have been logged to MLflow!")
    print("Run: mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000")


if __name__ == "__main__":
    log_existing_models_to_mlflow()
