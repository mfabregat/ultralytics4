import cv2
import os
import numpy as np
from pathlib import Path
from typing import Union, Optional


def imread(path: Union[str, Path], depth_dir: str = "depth") -> np.ndarray:
    """
    Read an RGB image and its corresponding depth image, then merge them together.
    
    Args:
        path (Union[str, Path]): Path to the RGB image file
        depth_dir (str, optional): Name of the directory containing depth images. Defaults to "depth".
        
    Returns:
        np.ndarray: Merged RGBD image as a numpy array
        
    Raises:
        FileNotFoundError: If either RGB or depth image does not exist
        ValueError: If there's an error reading or merging the images
    """
    # Convert path to Path object for cross-platform compatibility
    image_path = Path(path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"RGB image not found at {image_path}")
        
    # Construct the depth image path
    depth_path = image_path.parent.parent / depth_dir / (image_path.stem + '.png')
    
    # Read the RGB image
    rgb_image = cv2.imread(str(image_path))
    if rgb_image is None:
        raise ValueError(f"Failed to read RGB image at {image_path}")
    
    # Check if depth image exists
    if not depth_path.exists():
        raise FileNotFoundError(f"Depth image not found at {depth_path}")
    
    # Read the depth image
    depth_image = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_image is None:
        raise ValueError(f"Failed to read depth image at {depth_path}")
    
    # Reshape depth image if necessary
    if len(depth_image.shape) == 2:  # Single channel
        # Convert to 3D array with single channel for merging
        depth_image = depth_image[..., np.newaxis]
    
    # Merge the images
    try:
        merged_image = np.concatenate((rgb_image, depth_image), axis=2)
        return merged_image
    except Exception as e:
        raise ValueError(f"Error merging images: {str(e)}. RGB shape: {rgb_image.shape}, Depth shape: {depth_image.shape}")
