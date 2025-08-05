import argparse
import os
from PIL import Image
from instance_mapper import color_instances_from_sobel, plot_sobel_gradients
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

def main():
    args = parse_args()
    pred2D_names = os.listdir(args.pred_2D)
    time_steps = ["01", "02"]
    image_nums = [range(131,150), range(61,80)]
    slice_nums = range(0,59)

    for i, time_step in enumerate(time_steps):
        dir_name = f"{time_step}_3D_pred"
        grad_dir_name = f"{time_step}_3D_gradient"
        uniform_dir_name = f"{time_step}_3D_uniform"

        os.makedirs(os.path.join(args.root_directory, dir_name))
        for image_num in image_nums[i]:
            if image_num<10:
                image_num = "00"+str(image_num)
            elif image_num < 100:
                image_num = "0" + str(image_num)
            else:
                image_num = str(image_num)

            current_image = []
            current_gradients = []
            current_uniform_image = []

            for slice_num in slice_nums:
                slice_name = f"{dir_name}_man_seg{image_num}_slice_{slice_num}"
                slice_path = os.path.join(args.pred_2D, slice_name)

                slice = np.array(Image.open(slice_path))
                current_image.append(slice)

                gradients = plot_sobel_gradients(slice)
                current_gradients.append(gradients)

                uniform_prediction = color_instances_from_sobel(gradients)
                current_uniform_image.append(uniform_prediction)
            
            pred_3D_name = f"mask{image_num}.tif"
            tiff.imwrite(os.path.join(dir_name, pred_3D_name), np.array(current_image))
            tiff.imwrite(os.path.join(grad_dir_name, pred_3D_name), np.array(current_gradients))
            tiff.imwrite(os.path.join(uniform_dir_name, pred_3D_name), np.array(current_uniform_image))


if __name__ == "__main__":
    main()