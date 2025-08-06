import argparse
import os
from PIL import Image
from utils import color_instances_from_sobel, plot_sobel_gradients, remove_noise, extract_mask
from metric_calculation import calculate_iou 
import numpy as np
import tifffile as tiff

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "pred_2D",
        type= str,
        required=True,
        default="C:\\Users\\hasee\\Desktop\\DFKI\\Results\\Fluo-N3DH-SIM+\\2DResults\\stable_diffusion_fluo_with_bg_6e_6_Vanillawith0HorizontalFlip\\Epoch-24"
    )

    parser.add_argument(
        "--root_directory",
        type=str,
        required=True,
        default="C:\\Users\\hasee\\Desktop\\DFKI\\codes\\evalSoftware\\testing_dataset"
    )

    args = parser.parse_args()
    return args

def transform_slice(slice : np.array) -> np.array:
    """
    Transform a 2D image slice by applying Sobel gradient detection, coloring instances, and removing noise.

    Args:
        slice (np.array): A RGB 2D NumPy array representing the image slice to be processed.

    Returns:
        np.array: The processed image slice with noise removed after gradient and instance coloring steps.
    """
    gradients = plot_sobel_gradients(slice=slice)
    uniform_predictions = color_instances_from_sobel(sobel_image=gradients)
    noise_removed = remove_noise(slice=uniform_predictions)
    return noise_removed

def process_2D_slices(time_step : str, image_num: str, num_slices:int=59) -> np.array:
    '''
        Process 2D images by taking in directory name, image number and number of slices in the image

        Args:
            time_step (str) : time step of the image e.g 01
            image_num (str) : image number e.g 061
            num_slices (int) : number of slices in the image
        
        Returns:
            np.array [D,H,W]: 3D image in gray scale where there is no intra-instance inter-slice similarity
    '''
    current_image = []
    for slice_num in range(num_slices):
        slice_name = f"{time_step}_man_seg{image_num}_slice_{slice_num}"
        slice_path = os.path.join(args.pred_2D, slice_name)

        slice = np.array(Image.open(slice_path))
        processed_slice = transform_slice(slice=slice)
        current_image.append(processed_slice)

    return current_image

def process_3D_slices(threeDimImage: np.array) -> np.array:
    '''
    Provides inter-slice uniformity to 3D gray scale images
    
    Args:
        threeDimImage (np.ndarray): 3D gray scale image with intra-slice uniformity but no inter-slice uniformity [D,H,W]

    Returns:
        np.ndarray: 3D gray scale image with inter slice-uniformity 
    '''
    previous_slice = {}  # {instance_id: instance_mask}
    global_instances = {}  # {global_instance_id: {slice_no: slice_instance_id}}
    global_instance_id = 1  # Start from 1 (0 is background)
    
    for slice_num, slice in enumerate(threeDimImage):
        slice_instances = np.unique(slice)
        # Remove background (assuming 0 is background)
        slice_instances = slice_instances[slice_instances != 0]
        
        current_slice_masks = {}  # Store masks for current slice
        
        for slice_instance in slice_instances:
            instance_mask = extract_mask(img=slice, id=slice_instance)
            current_slice_masks[slice_instance] = instance_mask
            
            if slice_num == 0:
                # First slice: assign global IDs directly
                global_instances[global_instance_id] = {slice_num: slice_instance}
                global_instance_id += 1
            else:
                # Find best match with previous slice
                greatest_match = (-1, -1)  # (prev_instance_id, iou)
                
                for prev_slice_id, prev_slice_mask in previous_slice.items():
                    if prev_slice_id == 0:  # Skip background
                        continue
                    
                    iou = calculate_iou(pred_mask=prev_slice_mask, gt_mask=instance_mask)
                    
                    if iou > greatest_match[1]:
                        greatest_match = (prev_slice_id, iou)
                
                if greatest_match[1] > 0:
                    # Found a match - find the corresponding global ID
                    matched_global_id = None
                    for global_id, slice_mappings in global_instances.items():
                        if (slice_num - 1) in slice_mappings:
                            if slice_mappings[slice_num - 1] == greatest_match[0]:
                                # Add current instance to this global mapping
                                slice_mappings[slice_num] = slice_instance
                                matched_global_id = global_id
                                break
                    
                    if matched_global_id is None:
                        print(f"Warning: Could not find global mapping for matched instance {greatest_match[0]} in slice {slice_num-1}")
                        # Create new global instance as fallback
                        global_instances[global_instance_id] = {slice_num: slice_instance}
                        global_instance_id += 1
                else:
                    # No match found - create new global instance
                    global_instances[global_instance_id] = {slice_num: slice_instance}
                    global_instance_id += 1
        
        # Update previous_slice for next iteration
        previous_slice = current_slice_masks.copy()
    
    # Create remapped image
    remapped_image = np.zeros_like(threeDimImage, dtype=threeDimImage.dtype)

    # Invert the global_instances dictionary for easier lookup:
    # {(slice_no, slice_instance_id): global_instance_id}
    reverse_global_map = {}
    for global_id, slice_mappings in global_instances.items():
        for slice_no, slice_instance_id in slice_mappings.items():
            reverse_global_map[(slice_no, slice_instance_id)] = global_id

    # Apply the global mapping to create remapped image
    for slice_idx, slice in enumerate(threeDimImage):
        unique_instances_in_slice = np.unique(slice)

        for local_instance_id in unique_instances_in_slice:
            if local_instance_id == 0:  # Skip background
                continue

            key = (slice_idx, local_instance_id)
            if key in reverse_global_map:
                global_id = reverse_global_map[key]
                remapped_image[slice_idx][slice == local_instance_id] = global_id
            else:
                print(f"Warning: instance {local_instance_id} in slice {slice_idx} not found in global mapping")
    
    return remapped_image



def main():
    args = parse_args()
    time_steps = ["01", "02"]
    image_nums = [range(131,150), range(61,80)]

    for i, time_step in enumerate(time_steps):
        dir_name = f"{time_step}_3D_pred"

        os.makedirs(os.path.join(args.root_directory, dir_name))
        for image_num in image_nums[i]:
            if image_num<10:
                image_num = "00"+str(image_num)
            elif image_num < 100:
                image_num = "0" + str(image_num)
            else:
                image_num = str(image_num)

            intra_slice_uniform = process_2D_slices(time_step=time_step,image_num=image_num,num_slices=59)
            inter_slice_uniform = process_3D_slices(threeDimImage=intra_slice_uniform)
            
            pred_3D_name = f"mask{image_num}.tif"
            tiff.imwrite(os.path.join(dir_name, pred_3D_name), np.array(inter_slice_uniform))

if __name__ == "__main__":
    main()