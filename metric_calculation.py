import numpy as np
from sklearn.metrics import jaccard_score
from utils import extract_mask
import os
import argparse
from scipy.ndimage import find_objects
import tifffile as tif
from tqdm import tqdm

def segAccuracy(pred: np.array, gt: np.array, threshold: float) -> float:
    """
    Calculate segmentation accuracy at a specified IoU threshold for 3D grayscale predictions.
    Args:
        pred (np.array): 3D array containing predicted instance labels. Background is labeled as 0.
        gt (np.array): 3D array containing ground truth instance labels. Background is labeled as 0.
        threshold (float): IoU threshold for considering a prediction as a true positive.
    Returns:
        float: Segmentation accuracy, defined as the ratio of true positive predictions to the number of ground truth instances.
    Notes:
        - Each unique non-zero value in `pred` and `gt` is treated as a separate instance.
        - The function assumes the existence of `extract_mask` helper function.
    """
    # Input validation
    if pred.shape != gt.shape:
        raise ValueError(f"Prediction and ground truth shapes don't match: {pred.shape} vs {gt.shape}")
    
    if not (0 <= threshold <= 1):
        raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")
    
    Tp = 0
    
    pred_instance_ids = np.unique(pred).copy()  # Make a copy to avoid modification issues
    gt_instance_ids = np.unique(gt)
    
    # Remove background from gt count early
    gt_foreground_count = len(gt_instance_ids) - (1 if 0 in gt_instance_ids else 0)
    
    # Handle edge case where no ground truth instances exist
    if gt_foreground_count == 0:
        return 0.0
    
    # Pre-compute all prediction masks for efficiency
    pred_masks = {}
    for pred_instance_id in pred_instance_ids:
        if pred_instance_id != 0:  # Skip background
            pred_instance_mask = extract_mask(pred, pred_instance_id)
            pred_masks[pred_instance_id] = pred_instance_mask
    
    # Track which predictions have been matched to avoid double-counting
    matched_pred_ids = set()
    

    for gt_instance_id in gt_instance_ids:
        if gt_instance_id == 0:  # Skip background
            continue
            
        gt_instance_mask = extract_mask(gt, gt_instance_id)
        
        # Find best matching prediction
        best_match = None
        best_iou = -1

        print(f"Pred Instance IDs: {pred_instance_ids}")
        
        for pred_instance_id in pred_masks:
            if pred_instance_id in matched_pred_ids:  # Skip already matched predictions
                continue
                
            pred_instance_mask = pred_masks[pred_instance_id]
            
            try:
                iou = calculate_iou(pred_mask=pred_instance_mask, gt_mask=gt_instance_mask)
                
                if iou > best_iou:
                    best_iou = iou
                    best_match = pred_instance_id
            except Exception as e:
                print(f"Warning: Error calculating IoU for pred_id {pred_instance_id}, gt_id {gt_instance_id}: {e}")
                continue
        
        # Check if best match exceeds threshold
        if best_match is not None and best_iou >= threshold:
            Tp += 1
            matched_pred_ids.add(best_match)  # Mark as matched
    
    print(f"True Positives: {Tp}, Ground Truth Instances: {gt_foreground_count}")
    
    return Tp / gt_foreground_count

def calculate_iou(pred_mask: np.array, gt_mask: np.array) -> float:
    """
    Calculates the Intersection over Union (IoU) score between a predicted mask and a ground truth mask.
    Args:
        pred_mask (np.array): The predicted binary mask as a NumPy array.
        gt_mask (np.array): The ground truth binary mask as a NumPy array.
    Returns:
        float: The IoU (Jaccard index) score between the predicted and ground truth masks.
    Notes:
        Both input masks should be of the same shape and contain binary values (0 or 1).
    """
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(f"Mask shapes don't match: {pred_mask.shape} vs {gt_mask.shape}")
    
    # Convert to binary if not already
    pred_binary = (pred_mask > 0).astype(int)
    gt_binary = (gt_mask > 0).astype(int)
    
    # Handle edge case where both masks are empty
    if np.sum(pred_binary) == 0 and np.sum(gt_binary) == 0:
        return 1.0  # Perfect match for empty masks
    
    # Handle edge case where one mask is empty
    if np.sum(pred_binary) == 0 or np.sum(gt_binary) == 0:
        return 0.0
    
    # Flatten for jaccard_score
    gt_mask_flat = gt_binary.flatten()
    pred_mask_flat = pred_binary.flatten()
    
    try:
        iou = jaccard_score(gt_mask_flat, pred_mask_flat, zero_division=0)
        return iou
    except Exception as e:
        print(f"Warning: jaccard_score failed: {e}")
        return 0.0


def segAccuracyv2(pred: np.array, gt: np.array, threshold: float) -> float:
    """
    Optimized segmentation accuracy calculation at a specified IoU threshold for 3D grayscale predictions.
    
    Key optimizations:
    - Uses scipy.ndimage.find_objects for efficient bounding box extraction
    - Computes IoU only on bounding box regions to reduce memory usage
    - Avoids pre-computing all masks
    - Uses vectorized operations for IoU calculation
    
    Args:
        pred (np.array): 3D array containing predicted instance labels. Background is labeled as 0.
        gt (np.array): 3D array containing ground truth instance labels. Background is labeled as 0.
        threshold (float): IoU threshold for considering a prediction as a true positive.
    
    Returns:
        float: Segmentation accuracy, defined as the ratio of true positive predictions 
               to the number of ground truth instances.
    """
    # Input validation
    if pred.shape != gt.shape:
        raise ValueError(f"Prediction and ground truth shapes don't match: {pred.shape} vs {gt.shape}")
    
    if not (0 <= threshold <= 1):
        raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")
    
    # Get unique instance IDs (excluding background)
    gt_instance_ids = np.unique(gt)
    gt_instance_ids = gt_instance_ids[gt_instance_ids != 0]
    
    # Handle edge case where no ground truth instances exist
    if len(gt_instance_ids) == 0:
        return 0.0
    
    pred_instance_ids = np.unique(pred)
    pred_instance_ids = pred_instance_ids[pred_instance_ids != 0]
    
    # If no predictions, accuracy is 0
    if len(pred_instance_ids) == 0:
        return 0.0
    
    # Use scipy's find_objects for efficient bounding box extraction
    gt_slices = find_objects(gt)
    pred_slices = find_objects(pred)
    
    tp_count = 0
    matched_pred_ids = set()
    
    for gt_id in gt_instance_ids:
        gt_slice = gt_slices[gt_id - 1]  # find_objects returns 0-indexed list
        
        if gt_slice is None:  # Skip if object not found
            continue
            
        # Extract ground truth mask in bounding box region
        gt_bbox = gt[gt_slice]
        gt_mask = (gt_bbox == gt_id)
        
        best_iou = 0.0
        best_pred_id = None
        
        for pred_id in pred_instance_ids:
            if pred_id in matched_pred_ids:
                continue
                
            pred_slice = pred_slices[pred_id - 1]
            if pred_slice is None:
                continue
            
            # Calculate IoU using optimized bounding box approach
            iou = calculate_iou_optimized(pred, gt, pred_id, gt_id, pred_slice, gt_slice)
            
            if iou > best_iou:
                best_iou = iou
                best_pred_id = pred_id
        
        # Check if best match exceeds threshold
        if best_pred_id is not None and best_iou >= threshold:
            tp_count += 1
            matched_pred_ids.add(best_pred_id)
    
    return tp_count / len(gt_instance_ids)


def calculate_iou_optimized(pred_vol, gt_vol, pred_id, gt_id, pred_slice, gt_slice):
    """
    Optimized IoU calculation using bounding box intersection.
    
    Args:
        pred_vol: Full prediction volume
        gt_vol: Full ground truth volume  
        pred_id: Prediction instance ID
        gt_id: Ground truth instance ID
        pred_slice: Bounding box slice for prediction
        gt_slice: Bounding box slice for ground truth
    
    Returns:
        float: IoU score
    """
    try:
        # Calculate intersection of bounding boxes
        intersect_slice = tuple(
            slice(max(p.start, g.start), min(p.stop, g.stop))
            for p, g in zip(pred_slice, gt_slice)
        )
        
        # Check if bounding boxes intersect
        if any(s.start >= s.stop for s in intersect_slice):
            return 0.0
        
        # Extract masks in intersection region
        pred_region = pred_vol[intersect_slice]
        gt_region = gt_vol[intersect_slice]
        
        pred_mask = (pred_region == pred_id)
        gt_mask = (gt_region == gt_id)
        
        # Calculate intersection
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        
        if intersection == 0:
            return 0.0
        
        # Calculate union using inclusion-exclusion principle
        # Union = |A| + |B| - |A ∩ B|
        pred_size = (pred_vol == pred_id).sum()
        gt_size = (gt_vol == gt_id).sum()
        union = pred_size + gt_size - intersection
        
        return intersection / union if union > 0 else 0.0
        
    except Exception as e:
        warnings.warn(f"Error calculating IoU for pred_id {pred_id}, gt_id {gt_id}: {e}")
        return 0.0
