import argparse
import tifffile as tiff
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import label2rgb
from skimage.measure import label
from skimage.measure import label, regionprops

def color_instances_from_sobel(sobel_image: np.ndarray, threshold: int = 50) -> np.ndarray:
    """
    Detect and uniquely color instances from a Sobel-filtered image.
    
    Args:
        sobel_image (np.ndarray): 2D image (grayscale) after applying Sobel filtering.
        threshold (int): Threshold to binarize the Sobel image. Default is 50.
        
    Returns:
        np.ndarray: RGB image with uniquely colored instances.
    """
    # Threshold to create binary edge image
    _, binary = cv2.threshold(sobel_image, threshold, 255, cv2.THRESH_BINARY)

    # Invert binary image to prepare for filling (objects = white)
    inverted = cv2.bitwise_not(binary)

    # Morphological closing to fill gaps in edges
    filled = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # Label connected components
    labels = label(filled)

    # Get unique labels and their counts
    unique_labels, counts = np.unique(labels, return_counts=True)

    # Find the label with the maximum count (most repeated)
    most_repeated_label = unique_labels[np.argmax(counts)]

    # Create a copy of labels to modify
    labels_corrected = labels.copy()

    # If the most repeated label is not already 0, swap it with 0
    if most_repeated_label != 0:
        # Find where the current label 0 is (if it exists)
        mask_zero = (labels == 0)
        mask_most_repeated = (labels == most_repeated_label)
        
        # Swap: most repeated label becomes 0, and old 0 becomes the most repeated label
        labels_corrected[mask_most_repeated] = 0
        # labels_corrected[mask_zero] = most_repeated_label

    return labels_corrected

#@print_param_shapes
def plot_sobel_gradients(slice):

    image = cv2.cvtColor(slice, cv2.COLOR_RGB2BGR)
    image = slice
    gradients = {}

    for i, color in enumerate(['B', 'G', 'R']):
        channel = image[:, :, i]
        grad_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradients[color] = magnitude

    total_gradient = np.sqrt(
        gradients['R']**2 + gradients['G']**2 + gradients['B']**2
    )
    return total_gradient

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--destination_directory",
        type=str,
        default="C:\\Users\\hasee\\Desktop\\DFKI\\codes\\postProcessing\\test.tif"
    )

    parser.add_argument(
        "-s",
        "--source_directory",
        type=str,
        default="C:\\Users\\hasee\\Desktop\\DFKI\\Visual Results\\sobel vs contours vs threshold\\man_seg131_predictioni_vanila_model.tif"
    )

    args = parser.parse_args()
    return args



def main():
    args = parse_args()
    img = tiff.imread(args.source_directory)

    processed_img = []
    for slice_no, slice in enumerate(img):
        #slice is (h,w,3)
        gradients = plot_sobel_gradients(slice)
        uniform_prediction = color_instances_from_sobel(gradients)
        processed_img.append(uniform_prediction)
        
    processed_img = np.array(processed_img)
    print(f"{processed_img.shape=}")
    tiff.imwrite(args.destination_directory, processed_img)

    
    

if "__main__" == __name__:
    main()