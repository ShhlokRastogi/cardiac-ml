import os
import joblib
import numpy as np
import torch
from src.config import DEVICE, STAGE1_WEIGHTS_PATH, STAGE2_WEIGHTS_PATH, FEATURE_COLS
from src.models import AttentionUNet
from src.post_process import keep_largest_connected_component_3d
from src.data_prep import extract_clinical_feature_row

class CardiacDiagnosisPipeline:
    def __init__(self, stage1_weights=STAGE1_WEIGHTS_PATH, stage2_weights=STAGE2_WEIGHTS_PATH, device=DEVICE):
        self.device = device
        self.stage1_model = AttentionUNet(n_channels=1, n_classes=4, bilinear=False).to(self.device)
        if os.path.exists(stage1_weights):
            self.stage1_model.load_state_dict(torch.load(stage1_weights, map_location=self.device))
            self.stage1_model.eval()

        if os.path.exists(stage2_weights):
            self.stage2_classifier = joblib.load(stage2_weights)
        else:
            self.stage2_classifier = None

    @torch.no_grad()
    def predict_segmentation_3d(self, volume_2d_stack):
        self.stage1_model.eval()
        num_slices = volume_2d_stack.shape[2]
        pred_slices = []
        for slice_idx in range(num_slices):
            slice_2d = volume_2d_stack[:, :, slice_idx]
            tensor_in = torch.tensor(slice_2d, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            logits = self.stage1_model(tensor_in)
            pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
            pred_slices.append(pred_mask)
            
        raw_mask_3d = np.stack(pred_slices, axis=2).astype(np.uint8)
        clean_mask_3d = keep_largest_connected_component_3d(raw_mask_3d, classes=[1, 2, 3])
        return clean_mask_3d

    def predict_patient_end_to_end(self, patient_id, info_dict, ed_vol, es_vol):
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
