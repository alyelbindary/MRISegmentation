import re
import os
import pandas as pd
import numpy as np
import io
import cv2
import matplotlib.pyplot as plt
import datasets as dts
import imageio
import nrrd

from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from data_handling_functions import show_mask
from data_handling_functions import load_and_convert_mask
from data_handling_functions import pad_mask_and_img
from data_handling_functions import load_and_convert_image
from skimage import measure

def remove_small_blobs(mask, min_area_fraction=0.01, plot=False):
    """
    Remove small connected components in a 2D binary mask.

    Parameters
    ----------
    mask : np.ndarray
        2D binary mask containing blobs (0 and 1).
    min_area_fraction : float
        Blobs smaller than this fraction of the total mask area will be removed.
        Example: 0.01 → removes blobs smaller than 1% of the total foreground area.
    plot : bool
        If True, shows before/after visualization.

    Returns
    -------
    cleaned_mask : np.ndarray
        The mask after removing tiny blobs.
    """

    if mask.dtype != np.uint8 and mask.dtype != bool:
        mask = mask.astype(np.uint8)

    # Total area of mask (foreground pixels)
    total_area = np.sum(mask)
    if total_area == 0:
        # print("Warning: Mask is empty.")
        return mask

    min_area = total_area * min_area_fraction

    # Label connected components
    labeled_mask = measure.label(mask, connectivity=2)
    props = measure.regionprops(labeled_mask)

    # Build output mask
    cleaned_mask = np.zeros_like(mask)

    for prop in props:
        if prop.area >= min_area:
            cleaned_mask[labeled_mask == prop.label] = 1

    # Visualization
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(mask, cmap='gray')
        ax[0].set_title("Original Mask")
        ax[0].axis('off')

        ax[1].imshow(cleaned_mask, cmap='gray')
        ax[1].set_title(f"Cleaned Mask\n(min_area_fraction={min_area_fraction})")
        ax[1].axis('off')

        plt.tight_layout()
        plt.show()

    return cleaned_mask


def get_bounding_box(ground_truth_map):
    '''
    This function creates varying bounding box coordinates based on the segmentation contours as prompt for the SAM model
    The padding is random int values between 5 and 20 pixels
    '''

    if len(np.unique(ground_truth_map)) > 1:

        # get bounding box from mask
        y_indices, x_indices = np.where(ground_truth_map > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        # add perturbation to bounding box coordinates
        H, W = ground_truth_map.shape
        x_min = max(0, x_min - np.random.randint(5, 20))
        x_max = min(W, x_max + np.random.randint(5, 20))
        y_min = max(0, y_min - np.random.randint(5, 20))
        y_max = min(H, y_max + np.random.randint(5, 20))

        bbox = [x_min, y_min, x_max, y_max]

        return bbox
    else:
        return [0, 0, 256, 256]
        # return [0, 256]

class SAMDataset(Dataset):
    def __init__(self, dataset, target_height, target_width, processor, sam_version=1):
        """
        Dataset class for SAM1 and SAM2 models.

        Parameters
        ----------
        dataset : list of dicts or pd.DataFrame
            Must contain 'image_path' and 'mask_path' columns.
        target_height : int
            Height to pad images/masks to.
        target_width : int
            Width to pad images/masks to.
        processor : transformers processor
            SAM1 or SAM2 processor.
        sam_version : int
            1 for SAM1, 2 for SAM2 (determines how inputs are passed).
        """
        self.dataset = dataset
        self.processor = processor
        self.target_height = target_height
        self.target_width = target_width
        self.sam_version = sam_version

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image_path = item["image_path"]
        mask_path = item["mask_path"]

        # Load image and mask
        image = np.array(load_and_convert_mask(image_path))  # single channel
        ground_truth_mask = np.array(load_and_convert_mask(mask_path))
        ground_truth_mask[ground_truth_mask != 0] = 1  # binary

        # If mask has channel dimension, reduce
        if ground_truth_mask.ndim == 3:
            ground_truth_mask = ground_truth_mask[..., 0]

        # Pad to target size
        image, ground_truth_mask = pad_mask_and_img(
            image, ground_truth_mask,
            self.target_height, self.target_width
        )

        # Get bounding box prompt
        prompt = get_bounding_box(ground_truth_mask)

        # Prepare processor input based on SAM version
        if self.sam_version == 1:
            # SAM1 accepts integer bounding boxes
            input_boxes = [[prompt]]
            inputs = self.processor(
                image,
                input_boxes=input_boxes,
                return_tensors="pt"
            )

        elif self.sam_version == 2:
            # SAM2 (Meta)
            # No processor — return raw numpy arrays directly
            # to be used by SAM2ImagePredictor
            # image = image.astype(np.float32) / 255.0  # normalize to [0,1]
            sample = {
                "pixel_values": image,           # raw image, HxWx3
                "input_boxes": np.array(prompt),   # bounding box coords
                "ground_truth_mask": ground_truth_mask        # binary mask
            }
            return sample


        else:
            raise ValueError(f"Invalid sam_version: {self.sam_version}")

        # inputs = self.processor(
        #     image,
        #     input_boxes=input_boxes,
        #     return_tensors="pt"
        # )

        # Remove batch dimension added by processor
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # Add extra info for visualization or debugging
        inputs["ground_truth_mask"] = ground_truth_mask
        inputs["filename"] = image_path

        return inputs

    def display_img_and_msk(self, idx):
        """
        Display the image and ground truth mask at a given index.
        """
        item = self.dataset[idx]

        # Load image
        img = np.array(load_and_convert_image(item["image_path"]))

        # Load mask and convert to binary
        ground_truth_seg = np.array(load_and_convert_mask(item["mask_path"]))
        ground_truth_seg[ground_truth_seg != 0] = 1
        ground_truth_seg = ground_truth_seg * 255

        # If mask has channel dimension, reduce
        if ground_truth_seg.ndim == 3:
            ground_truth_seg = ground_truth_seg[..., 0]

        # Display shapes
        print(f"img shape : {img.shape}")
        print(f"mask shape : {ground_truth_seg.shape}")

        # Display
        fig, axes = plt.subplots()
        axes.imshow(img)
        show_mask(ground_truth_seg, axes, random_color=False)
        axes.title.set_text("Ground truth mask")
        axes.axis("off")

class SAM2Dataset(Dataset):
    def __init__(self, dataset, target_height, target_width, processor):
        """
        Dataset class for SAM1 and SAM2 models.

        Parameters
        ----------
        dataset : list of dicts or pd.DataFrame
            Must contain 'image_path' and 'mask_path' columns.
        target_height : int
            Height to pad images/masks to.
        target_width : int
            Width to pad images/masks to.
        processor : transformers processor
            SAM1 or SAM2 processor.
        sam_version : int
            1 for SAM1, 2 for SAM2 (determines how inputs are passed).
        """
        self.dataset = dataset
        self.processor = processor
        self.target_height = target_height
        self.target_width = target_width

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image_path = item["image_path"]
        mask_path = item["mask_path"]

        # Load image and mask
        image = np.array(load_and_convert_mask(image_path))  # single channel
        ground_truth_mask = np.array(load_and_convert_mask(mask_path))
        ground_truth_mask[ground_truth_mask != 0] = 1  # binary

        # If mask has channel dimension, reduce
        if ground_truth_mask.ndim == 3:
            ground_truth_mask = ground_truth_mask[..., 0]

        # Pad to target size
        image, ground_truth_mask = pad_mask_and_img(
            image, ground_truth_mask,
            self.target_height, self.target_width
        )

        # Get bounding box prompt
        prompt = get_bounding_box(ground_truth_mask)

        input_boxes = [[prompt]]
        # Convert to Python ints
        input_boxes = [[[int(x) for x in box] for box in image_boxes] for image_boxes in input_boxes]
        # print(f"input boxes : \n{input_boxes}")
        inputs = self.processor(
            images=image,
            input_boxes=input_boxes,
            return_tensors="pt"
        )

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # Add extra info for visualization or debugging
        inputs["ground_truth_mask"] = ground_truth_mask
        inputs["filename"] = image_path

        return inputs

    def display_img_and_msk(self, idx):
        """
        Display the image and ground truth mask at a given index.
        """
        item = self.dataset[idx]

        # Load image
        img = np.array(load_and_convert_image(item["image_path"]))

        # Load mask and convert to binary
        ground_truth_seg = np.array(load_and_convert_mask(item["mask_path"]))
        ground_truth_seg[ground_truth_seg != 0] = 1
        ground_truth_seg = ground_truth_seg * 255

        # If mask has channel dimension, reduce
        if ground_truth_seg.ndim == 3:
            ground_truth_seg = ground_truth_seg[..., 0]

        # Display shapes
        print(f"img shape : {img.shape}")
        print(f"mask shape : {ground_truth_seg.shape}")

        # Display
        fig, axes = plt.subplots()
        axes.imshow(img)
        show_mask(ground_truth_seg, axes, random_color=False)
        axes.title.set_text("Ground truth mask")
        axes.axis("off")

class MultiSAMDataset(Dataset):
    def __init__(self, dataset, target_height, target_width, processor):
        """
        dataset : DataFrame with columns:
            - water_path
            - fat_path
            - mask_path
        """
        self.dataset = dataset
        self.processor = processor
        self.target_height = target_height
        self.target_width = target_width

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        water_path = item["water_path"]
        fat_path = item["fat_path"]
        mask_path = item["mask_path"]

        # ----------------------------------------------------
        # Load images (single-channel MRI)
        # ----------------------------------------------------
        water_img = np.array(load_and_convert_mask(water_path))
        fat_img   = np.array(load_and_convert_mask(fat_path))

        # ----------------------------------------------------
        # Load and preprocess ground truth mask
        # ----------------------------------------------------
        gt_mask = np.array(load_and_convert_mask(mask_path))
        gt_mask[gt_mask != 0] = 1  # binary mask

        if gt_mask.ndim == 3:
            gt_mask = gt_mask[..., 0]

        # ----------------------------------------------------
        # Pad all modalities to target resolution
        # ----------------------------------------------------
        water_img, gt_mask_padded = pad_mask_and_img(
            water_img, gt_mask,
            self.target_height, self.target_width
        )

        fat_img, _ = pad_mask_and_img(
            fat_img, gt_mask,
            self.target_height, self.target_width
        )

        # ----------------------------------------------------
        # Compute bounding-box prompt from padded GT mask
        # ----------------------------------------------------
        prompt = [0, 0, self.target_width, self.target_height]

        input_boxes = [[prompt]]
        input_boxes = [[[int(x) for x in box] for box in img_boxes]
                       for img_boxes in input_boxes]

        # ----------------------------------------------------
        # Process water modality
        # ----------------------------------------------------
        inputs_water = self.processor(
            images=water_img,
            input_boxes=input_boxes,
            return_tensors="pt"
        )
        inputs_water = {k: v.squeeze(0) for k, v in inputs_water.items()}

        # ----------------------------------------------------
        # Process fat modality
        # ----------------------------------------------------
        inputs_fat = self.processor(
            images=fat_img,
            input_boxes=input_boxes,
            return_tensors="pt"
        )
        inputs_fat = {k: v.squeeze(0) for k, v in inputs_fat.items()}

        # ----------------------------------------------------
        # Add GT mask and filename
        # ----------------------------------------------------
        out = {
            "water": inputs_water,
            "fat": inputs_fat,
            "ground_truth_mask": gt_mask_padded,
            "filename": water_path
        }

        return out
