import argparse
import tifffile as tiff
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import label2rgb
from skimage.measure import label
from skimage.measure import label, regionprops
import copy
from sklearn.metrics import jaccard_score

def color_instances_from_sobel(sobel_image: np.ndarray, threshold: int = 50) -> np.ndarray:
    """
    Labels and colors connected instances in a Sobel edge image.
    This function processes a Sobel edge-detected image to identify and label distinct connected regions (instances).
    It applies thresholding, morphological closing, and connected component labeling. The background label is set
    to the most frequent label in the image to ensure consistency.
    Args:                       
        sobel_image (np.ndarray): Input Sobel edge image as a 2D NumPy array.
        threshold (int, optional): Threshold value for binarizing the edge image. Defaults to 50.
    Returns:
        np.ndarray: A 2D array of the same shape as `sobel_image`, where each connected instance is assigned a unique label.
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
    """
    Computes the combined Sobel gradient magnitude for each color channel of an RGB image slice.
    This function calculates the Sobel gradients in both the x and y directions for each of the
    three color channels (Red, Green, Blue) of the input image. It then computes the gradient
    magnitude for each channel and combines them to produce a single gradient magnitude image.
    Args:
        slice (np.ndarray): An RGB image slice as a NumPy array of shape (H, W, 3).
    Returns:
        np.ndarray: A 2D array representing the combined gradient magnitude of the input image.
    """

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

def remove_noise(slice : np.array, threshold: int = 20) -> np.array:
    '''
        Remvoes noises from the image based on count of the instance

        Args:
            slice (np.array): 2D gray scale image representing a slice in 3D image s
            threshold (int): threshold for count of instance pixels 
    
        Returns:
            np.array : slice with noise removed through count thresholding
    '''
    instance_ids, counts = np.unique(slice, return_counts=True)
    noise_removed = copy.deepcopy(slice)
    for i, instance_id in enumerate(instance_ids):
        if counts[i] < threshold:
            noise_removed = np.where(noise_removed==instance_id, 0, slice)
    
    return noise_removed

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
        noise_removed = remove_noise(slice=uniform_prediction)
        processed_img.append(noise_removed)
        
    processed_img = np.array(processed_img)
    print(f"{processed_img.shape=}")
    tiff.imwrite(args.destination_directory, processed_img)

    
    

if "__main__" == __name__:
    main()