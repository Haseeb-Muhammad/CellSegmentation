import os
import argparse
from SEGAccuracy import segAccuracy
import tifffile as tif


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--root_directory",
        type=str,
        required=True,
        default="C:\\Users\\hasee\\Desktop\\DFKI\\datasets\\Fluo-N3DH-SIM+\\train")

    parser.add_argument(
        "--num_digits",
        type=int,
        required=False,
        default=1)
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args() 

    image_count=0
    SEGAccuracy=0
    
    for time in range(1,args.num_digits+1):
        gt_dir = os.path.join(args.root_directory, f"0{time}_GT", "SEG")
        pred_dir = os.path.join(args.root_directory, f"0{time}_RES")
        
        for gt_name in os.listdir(gt_dir):
            gt_path = os.path.join(gt_dir, gt_name)

            img_num = gt_name.split("seg")[1]
            pred_path = os.path.join(pred_dir, f"mask{img_num}")

            pred = tif.imread(pred_path)
            gt = tif.imread(gt_path)
            
            image_count+=1
            SEGAccuracy += segAccuracy(pred=pred, gt=gt)

    meanSEGAccuracy=SEGAccuracy / image_count
    print(f"{meanSEGAccuracy=}")


if __name__ == "__main__":
    main()