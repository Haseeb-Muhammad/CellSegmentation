
import numpy as np
from sklearn.metrics import jaccard_score

def extract_mask(img:np.array, id:int) -> np.array:
    """
    Extracts a binary mask from the input image where the pixels equal to the specified id are set to 1, and all others are set to 0.

    Args:
        img (np.array): Input image as a NumPy array.
        id (int): The pixel value to extract as a mask.

    Returns:
        np.array: A binary mask of the same shape as `img`, with 1 where `img` equals `id`, and 0 elsewhere.
    """
    return np.where(img==id, 1,0)

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
    
    gt_mask_flat = gt_mask.flatten()
    pred_mask_flat = pred_mask.flatten()

    iou = jaccard_score(gt_mask_flat, pred_mask_flat)
    return iou

def segAccuracy(pred:np.array, gt:np.array, threshold:int) -> float:
    """
    Calculate segmentation accuracy at a specified IoU threshold for 3D grayscale predictions.
    Args:
        pred (np.array): 3D array containing predicted instance labels. Background is labeled as 0.
        gt (np.array): 3D array containing ground truth instance labels. Background is labeled as 0.
        threshold (int): IoU threshold for considering a prediction as a true positive.
    Returns:
        float: Segmentation accuracy, defined as the ratio of true positive predictions to the number of ground truth instances.
    Notes:
        - Each unique non-zero value in `pred` and `gt` is treated as a separate instance.
        - The function assumes the existence of `extract_mask` and `calculate_iou` helper functions.
    """
    Tp = 0
    
    pred_instance_ids = np.unique(pred)
    gt_instance_ids = np.unique(gt)

    pred_masks = {}

    for pred_instance_id in pred_instance_ids:
        pred_instance_mask = extract_mask(pred, pred_instance_id)
        pred_masks[pred_instance_id] = pred_instance_mask


    for gt_instance_id in gt_instance_ids:
        if gt_instance_id == 0:
            continue
        gt_instance_mask = extract_mask(gt, gt_instance_id)
        
        # (pred_instance_id, iou)
        greatest_match = (-1,-1)
        for pred_instance_id in pred_instance_ids:
            if pred_instance_id == 0:
                continue

            pred_instance_mask = pred_masks[pred_instance_id]
            iou = calculate_iou(pred_mask=pred_instance_mask, gt_mask=gt_instance_mask)

            if iou>greatest_match:
                greatest_match = (pred_instance_id, iou)

        if greatest_match[1] >= threshold:
            Tp +=1
            del pred_masks[greatest_match[0]]
    
    return Tp / (len(gt_instance_ids)-1) #-1 to account for background in ground truth

