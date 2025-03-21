import glob
import os
from typing import Tuple

import numpy as np
from PIL import Image


def compute_mean_and_std(dir_name: str) -> Tuple[float, float]:
    """Compute the mean and the standard deviation of all images present within the directory.

    Note: convert the image in grayscale and then in [0,1] before computing mean
    and standard deviation

    Mean = (1/n) * \sum_{i=1}^n x_i
    Variance = (1 / (n-1)) * \sum_{i=1}^n (x_i - mean)^2
    Standard Deviation = sqrt( Variance )

    Args:
        dir_name: the path of the root dir

    Returns:
        mean: mean value of the gray color channel for the dataset
        std: standard deviation of the gray color channel for the dataset
    """
    mean = None
    std = None

    ############################################################################
    # Student code begin
    ############################################################################

    pixel_values = []
    means=[]
    stds=[]
   
    # Get all images from all subfolders
    image_paths = glob.glob(os.path.join(dir_name, "*", "*", "*.jpg"))
    
    for img_file in image_paths:
        if img_file.endswith(('.jpg', '.png', '.jpeg')):  # Ensure valid image format
            #print(img_file)
            img = Image.open(img_file).convert("L")  # Convert to grayscale
            img_array = np.array(img) / 255.0  # Normalize pixel values to [0,1]
            pixel_values.extend(img_array.flatten())  # Flatten to 1D list
     
    pixel_values = np.array(pixel_values)
    mean = np.mean(pixel_values)
    std = np.std(pixel_values, ddof=1)  # Using sample standard deviation
    
    """  raise NotImplementedError(
            "`compute_mean_and_std` function in "
            + "`stats_helper.py` needs to be implemented"
        ) """
   
    ############################################################################
    # Student code end
    ############################################################################
    return mean, std
