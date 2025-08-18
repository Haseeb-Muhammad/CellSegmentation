from metric_calculation import segAccuracyv2
import numpy as np
from sklearn.metrics import jaccard_score
from utils import extract_mask
import os
import argparse
import tifffile as tif
from tqdm import tqdm
def parse_args():
    parser = argparse.ArgumentParser(description="Calculate segmentation accuracy metrics")
    
    parser.add_argument(
        "--root_directory",
        type=str,
        required=False,
        default="/netscratch/muhammad/datasets/Fluo-N3DH-SIM+/Combined",
        help="Root directory containing the dataset")

    parser.add_argument(
        "--seq",
        type=int,
        required=False,
        default=1,
        help="Sequence number to process")

    parser.add_argument(
        "--num_digits",
        type=int,
        required=False,
        default=1,
        help="Number of digits in sequence naming (unused in current implementation)")
    
    parser.add_argument(
        "--threshold",
        type=float,
        required=False,
        default=0.5,
        help="IoU threshold for true positive classification")
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args() 
    
    image_count = 0
    SEGAccuracy = 0.0
    time = args.seq

    gt_dir = os.path.join(args.root_directory, f"0{time}_GT", "SEG")
    pred_dir = os.path.join(args.root_directory, f"0{time}_RES")
    
    
    gt_files = [f for f in os.listdir(gt_dir) if f.endswith(('.tif', '.tiff'))]
    
    print(f"Processing {len(gt_files)} images...")
    
    failed_images = []
    
    for gt_name in tqdm(gt_files, desc="Processing images"):
        gt_path = os.path.join(gt_dir, gt_name)

        img_num = gt_name.split("seg")[1]
        
        pred_path = os.path.join(pred_dir, f"mask{img_num}")
        
        pred = tif.imread(pred_path)
        gt = tif.imread(gt_path)

        image_count += 1
        accuracy = segAccuracyv2(pred=pred, gt=gt, threshold=args.threshold)
        SEGAccuracy += accuracy
        print(f"Image {gt_name}: accuracy = {accuracy:.4f}")
            
            
    meanSEGAccuracy = SEGAccuracy / image_count
    print(f"Mean Segmentation Accuracy: {meanSEGAccuracy:.4f}")

if __name__ == "__main__":
    main()