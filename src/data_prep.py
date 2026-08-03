import os
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
from src.config import VOXEL_SPACING

def parse_info_cfg(info_path):
    data = {}
    if not os.path.exists(info_path):
        return data
    with open(info_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            key, val = line.split(':', 1)
            data[key.strip()] = val.strip()
    return data

def calculate_bsa(height_cm, weight_kg):
    if height_cm <= 0 or weight_kg <= 0:
        return 1.75
    return 0.007184 * (height_cm ** 0.725) * (weight_kg ** 0.425)

def calculate_volume_ml(mask_3d, class_idx, voxel_spacing=VOXEL_SPACING):
    voxel_count = np.sum(mask_3d == class_idx)
    voxel_vol_ml = (voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2]) / 1000.0
    return float(voxel_count * voxel_vol_ml)

def extract_myo_thickness_mm(mask_3d, voxel_spacing=VOXEL_SPACING):
    max_thickness = 0.0
    num_slices = mask_3d.shape[2]
    for s in range(num_slices):
        myo_slice = (mask_3d[:, :, s] == 2)
        if not np.any(myo_slice):
            continue
        dt = ndimage.distance_transform_edt(myo_slice, sampling=(voxel_spacing[0], voxel_spacing[1]))
        if np.max(dt) > max_thickness:
            max_thickness = float(np.max(dt))
    return max_thickness * 2.0

def extract_clinical_feature_row(patient_id, info_dict, ed_mask, es_mask):
    height = float(info_dict.get("Height", 170.0))
    weight = float(info_dict.get("Weight", 70.0))
    bsa = calculate_bsa(height, weight)
    
    rv_edv = calculate_volume_ml(ed_mask, class_idx=1)
    rv_esv = calculate_volume_ml(es_mask, class_idx=1)
    rv_ef = ((rv_edv - rv_esv) / rv_edv * 100.0) if rv_edv > 0 else 0.0
    
    myo_edv = calculate_volume_ml(ed_mask, class_idx=2)
    lvm_g = myo_edv * 1.05
    
    lv_edv = calculate_volume_ml(ed_mask, class_idx=3)
    lv_esv = calculate_volume_ml(es_mask, class_idx=3)
    lv_ef = ((lv_edv - lv_esv) / lv_edv * 100.0) if lv_edv > 0 else 0.0
    
    max_thickness = extract_myo_thickness_mm(ed_mask)
    
    return {
        "Patient_ID": patient_id,
        "Group": info_dict.get("Group", "Unknown"),
        "Height": height,
        "Weight": weight,
        "BSA": bsa,
        "RVEDV": rv_edv,
        "RVESV": rv_esv,
        "RVEF": rv_ef,
        "RVEDVI": rv_edv / bsa,
        "RVESVI": rv_esv / bsa,
        "MYO_EDV": myo_edv,
        "LVM_g": lvm_g,
        "LVMI": lvm_g / bsa,
        "LVEDV": lv_edv,
        "LVESV": lv_esv,
        "LVEF": lv_ef,
        "LVEDVI": lv_edv / bsa,
        "LVESVI": lv_esv / bsa,
        "Max_MYO_Thickness_mm": max_thickness
    }
