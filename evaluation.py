import os
import argparse
from SEGAccuracy import segAccuracy
import tiffile as tif


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--pred_directory",
        type=str,
        required=True,
        default="")
    
    parser.add_argument(
        "--gt_directory",
        type=str,
        required=False,
        default="C:\\Users\\hasee\\Desktop\\DFKI\\datasets\\Fluo-N3DH-SIM+\\train")
    
def main():
    args = parse_args() 

    times = ["01", "02"]
    image_nums = [range(131,150), range(61,80)]
    
    image_count=0
    SEGAccuracy=0
    
    for i,time in enumerate(times):
        for j in image_nums[i]:
            if j<10:
                image_num = "00"+str(j)
            elif j<100:
                image_num = "0"+str(j)
            else:
                image_num = str(j)
            
            pred_path = os.path.join(args.pred_directory, f"man_seg{image_num}.tif")
            gt_path = os.path.join(args.gt_directory, f"{time}_GT", "SEG",f"man_seg{image_num}.tif")
            
            pred = tif.imread(pred_path)
            gt = tif.imread(gt_path)
            
            image_count+=1
            SEGAccuracy += segAccuracy(pred=pred, gt=gt)

    meanSEGAccuracy=SEGAccuracy / image_count
    return meanSEGAccuracy    


if __name__ == "__main__":
    main()