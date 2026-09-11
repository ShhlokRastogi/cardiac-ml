import os
import gc
import joblib
import numpy as np
import torch
from src.config import DEVICE, STAGE1_WEIGHTS_PATH, STAGE2_WEIGHTS_PATH, FEATURE_COLS
from src.models import AttentionUNet
from src.post_process import keep_largest_connected_component_3d
from src.data_prep import extract_clinical_feature_row

# Memory optimization for cloud environments (Render 512MB limit)
torch.set_num_threads(1)
if hasattr(torch, "set_num_interop_threads"):
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass


class CardiacDiagnosisPipeline:
    def __init__(self, stage1_weights=STAGE1_WEIGHTS_PATH, stage2_weights=STAGE2_WEIGHTS_PATH, device=DEVICE):
        self.device = device
        self.stage1_weights = stage1_weights
        self.stage2_weights = stage2_weights
        self.stage1_model = None
        self.stage2_classifier = None

    def _load_stage1_if_needed(self):
        if self.stage1_model is None:
            model = AttentionUNet(n_channels=1, n_classes=4, bilinear=False).to(self.device)
            if os.path.exists(self.stage1_weights):
                try:
                    with open(self.stage1_weights, "rb") as f:
                        header = f.read(7)
                    if header != b"version":
                        model.load_state_dict(
                            torch.load(self.stage1_weights, map_location=self.device, weights_only=False)
                        )
                        model.eval()
                except Exception as e:
                    print(f"Warning: Could not load Stage 1 weights ({e})")
            self.stage1_model = model

    def _load_stage2_if_needed(self):
        if self.stage2_classifier is None and os.path.exists(self.stage2_weights):
            try:
                with open(self.stage2_weights, "rb") as f:
                    header = f.read(7)
                if header != b"version":
                    self.stage2_classifier = joblib.load(self.stage2_weights)
            except Exception as e:
                print(f"Warning: Could not load Stage 2 weights ({e})")

    @torch.inference_mode()
    def predict_segmentation_3d(self, volume_2d_stack):
        self._load_stage1_if_needed()
        self.stage1_model.eval()

        num_slices = volume_2d_stack.shape[2]
        pred_slices = []

        for slice_idx in range(num_slices):
            slice_2d = volume_2d_stack[:, :, slice_idx]
            tensor_in = torch.from_numpy(slice_2d).unsqueeze(0).unsqueeze(0).to(self.device)
            logits = self.stage1_model(tensor_in)
            pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            pred_slices.append(pred_mask)

        raw_mask_3d = np.stack(pred_slices, axis=2)
        clean_mask_3d = keep_largest_connected_component_3d(raw_mask_3d, classes=[1, 2, 3])

        del pred_slices, raw_mask_3d
        gc.collect()
        return clean_mask_3d

    def predict_patient_end_to_end(self, patient_id, info_dict, ed_vol, es_vol):
        self._load_stage2_if_needed()

        ed_mask = self.predict_segmentation_3d(ed_vol)
        es_mask = self.predict_segmentation_3d(es_vol)
        feature_row = extract_clinical_feature_row(patient_id, info_dict, ed_mask, es_mask)
        x_feat = np.array([[feature_row[col] for col in FEATURE_COLS]], dtype=np.float64)

        if self.stage2_classifier is not None:
            pred_diagnosis = self.stage2_classifier.predict(x_feat)[0]
            probabilities = self.stage2_classifier.predict_proba(x_feat)[0]
            confidence = float(np.max(probabilities) * 100.0)
            class_probs = {cls: float(p) for cls, p in zip(self.stage2_classifier.classes_, probabilities)}
        else:
            pred_diagnosis = "Unknown"
            confidence = 0.0
            class_probs = {}

        gc.collect()
        return {
            "Patient_ID": patient_id,
            "True_Diagnosis": feature_row.get("Group", "Unknown"),
            "Predicted_Diagnosis": pred_diagnosis,
            "Confidence_Percentage": confidence,
            "Class_Probabilities": class_probs,
            "Clinical_Features": feature_row,
            "ED_Mask": ed_mask,
            "ES_Mask": es_mask
        }

    def predict_3d_volume(self, volume_2d_stack):
        return self.predict_segmentation_3d(volume_2d_stack)

    def extract_biometrics(self, patient_id, info_dict, ed_mask, es_mask):
        return extract_clinical_feature_row(patient_id, info_dict, ed_mask, es_mask)

    def classify_disease(self, feature_row):
        self._load_stage2_if_needed()
        x_feat = np.array([[feature_row[col] for col in FEATURE_COLS]], dtype=np.float64)
        if self.stage2_classifier is not None:
            pred_diagnosis = self.stage2_classifier.predict(x_feat)[0]
            probabilities = self.stage2_classifier.predict_proba(x_feat)[0]
            confidence = float(np.max(probabilities) * 100.0)
            class_probs = {cls: float(p) for cls, p in zip(self.stage2_classifier.classes_, probabilities)}
        else:
            pred_diagnosis = "Unknown"
            confidence = 0.0
            class_probs = {}
        return pred_diagnosis, confidence, class_probs
