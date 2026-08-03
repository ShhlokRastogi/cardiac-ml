import numpy as np
import scipy.ndimage as ndimage

def keep_largest_connected_component_3d(pred_volume_3d, classes=[1, 2, 3]):
    cleaned_volume = np.copy(pred_volume_3d)
    for cls in classes:
        binary_mask = (pred_volume_3d == cls)
        if not np.any(binary_mask):
            continue
        labeled_array, num_features = ndimage.label(binary_mask)
        if num_features <= 1:
            continue
        component_sizes = ndimage.sum(binary_mask, labeled_array, range(1, num_features + 1))
        largest_component_label = np.argmax(component_sizes) + 1
        smaller_components_mask = (labeled_array > 0) & (labeled_array != largest_component_label)
        cleaned_volume[smaller_components_mask & (cleaned_volume == cls)] = 0
    return cleaned_volume
