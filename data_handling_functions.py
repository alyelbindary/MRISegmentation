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

from tiff_to_jpgs import adjust_jpg

import numpy as np
import matplotlib.pyplot as plt
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

    not_binary = False
    mask_vals = np.unique(mask)
    if (len(mask_vals) > 1) :
        if (mask_vals[1] > 1) :
            not_binary = True
            mask = mask/mask_vals[1]

    if mask.dtype != np.uint8 and mask.dtype != bool:
        mask = mask.astype(np.uint8)

    # Total IMAGE area (not foreground area)
    image_area = mask.shape[0] * mask.shape[1]
    min_area = image_area * min_area_fraction

    # If mask is empty, nothing to clean
    if np.sum(mask) == 0:
        print("Warning: Mask is empty.")
        return mask

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

    if not_binary :
        cleaned_mask = cleaned_mask*255

    return cleaned_mask

def extract_week_from_folder(folder_name):
    """
    Extract numeric week index from folder like 'T0-00J' or 'T3-21J'.
    Returns integer (e.g., 'T3-21J' → 3)
    """
    match = re.search(r'T(\d+)', folder_name)
    return int(match.group(1)) if match else None


def get_id_and_mask_label(image_paths):
    """
    Extract image id (the middle digits) and mask flag from filenames.
    Example:
        '10_000_mask.png' → id=0, is_mask=1
        '10_000.png' → id=0, is_mask=0
    """
    ids, is_masks = [], []

    for path in image_paths:
        filename = os.path.basename(path)
        parts = filename.split('_')
        img_id = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else -1
        ids.append(img_id)

        is_mask = 1 if 'mask' in filename.lower() else 0
        is_masks.append(is_mask)

    return ids, is_masks


def build_df(image_paths):
    """
    Build dataframe directly from a list of image paths.
    Extracts subject, week, id, and mask info from filenames and directory structure.
    """
    subjects, weeks, ids, mask_labels = [], [], [], []

    for path in image_paths:
        parts = path.replace("\\", "/").split("/")  # normalize separators

        subject = parts[-4]   # e.g. "8.31"
        week = parts[-3]      # e.g. "T0-00J"
        is_mask = 1 if "mask" in parts[-2].lower() or "mask" in parts[-1].lower() else 0

        filename = os.path.basename(path)
        id_part = filename.split("_")[1]  # e.g. "000"
        ids.append(int(id_part))
        subjects.append(subject)
        weeks.append(week)
        mask_labels.append(is_mask)

    df_images = pd.DataFrame({
        "subject": subjects,
        "week": weeks,
        "id": ids,
        "image_path": image_paths,
        "is_mask": mask_labels,
    })

    return df_images

def _load(image_path, as_tensor=True):
    image = Image.open(image_path)
    return np.array(image).astype(np.float32) / 255.

def generate_label(mask_path, load_fn):
    mask = load_fn(mask_path)
    if mask.max() > 0:
        return 1 # Brain Tumor Present
    return 0 # Normal

def load_and_convert_mask(mask_path):
    try:
        if os.path.exists(mask_path):
            mask = Image.open(mask_path)
            return mask
        else:
            print(f"File not found: {mask_path}")
            return None
    except Exception as e:
        print(f"Error loading image {mask_path}: {e}")
        return None

def load_and_convert_image(image_path):
    try:
        if os.path.exists(image_path):
            with open(image_path, 'rb') as f:
                image = io.BytesIO(f.read())  # Read image data as bytes
            image = (Image.fromarray(adjust_jpg(np.array(Image.open(image))))).convert('RGB')  # Convert bytes to numpy array
            return image
        else:
            print(f"File not found: {image_path}")
            return None
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None
    
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

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

def get_max_dimensions(df):
    """
    Compute the maximum height and width across all images and masks in a DataFrame.
    Expects columns 'image_path' and 'mask_path'.
    """
    max_h, max_w = 0, 0

    for idx, row in df.iterrows():
        for key in ['image_path', 'mask_path']:
            path = row.get(key, None)
            if path is None or not isinstance(path, str) or path.strip() == "":
                continue  # skip missing paths

            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"[Warning] Could not read file: {path}")
                continue

            h, w = img.shape[:2]
            max_h = max(max_h, h)
            max_w = max(max_w, w)

    print(f"✅ Max height: {max_h}, Max width: {max_w}")
    return max_h, max_w

def pad_mask_and_img(img, mask, target_height, target_width):
    """
    Pads image and mask to (target_height, target_width)
    """
    h, w = mask.shape[:2]
    pad_h = max(0, target_height - h)
    pad_w = max(0, target_width - w)

    # Pad mask
    if mask.ndim == 2:
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    else:
        mask = np.pad(mask, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)

    # Pad image
    if img.ndim == 2:
        img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    else:
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)

    return img, mask

def adjust_brightness(image, brightness=70):
    # Convert image to HSV (Hue, Saturation, Value) format
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Increase the V (Value) channel to make the image brighter
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, brightness)
    v = np.clip(v, 0, 255)  # Ensure pixel values are within valid range
    final_hsv = cv2.merge((h, s, v))

    # Convert back to BGR format
    bright_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    return bright_image

# Function to draw a bounding box around the object in an image
def draw_bounding_box(image):

    # load_and_convert_image

    # Load the image
    image = cv2.imread(image)

    image = adjust_brightness(image)

    # Convert the image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a blur to the image to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # blurred = cv2.GaussianBlur(np.array(image), (5, 5), 0)

    # Threshold the image to get the object in binary
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour, assuming the object to bound is the largest one
    largest_contour = max(contours, key=cv2.contourArea)

    # Draw a bounding box around the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    plt.imshow(image)

    x_min = x
    x_max = x + w
    y_min = y
    y_max = y + h

    return [x_min, y_min, x_max, y_max]

def convert_nrrd_to_png_structure(input_folder="data_nrrd", output_base="new_data"):
    """
    Convert NRRD volumes (SEG, WATER, FAT) into PNG slices using structure:
      new_data/<subject>/<timepoint>-<day>/{Mask,Water,Fat}/<prefix>###_{mask|w|f}.png
    """

    timepoint_map = {
        "T0": "00J",
        "T1": "07J",
        "T2": "14J",
        "T3": "21J",
        "T4": "28J",
        "T5": "32J"
    }

    os.makedirs(output_base, exist_ok=True)
    nrrd_files = [f for f in os.listdir(input_folder) if f.endswith(".nrrd")]

    if not nrrd_files:
        print("⚠️ No NRRD files found in input folder.")
        return

    print(f"Found {len(nrrd_files)} NRRD files to process.\n")

    for filename in sorted(nrrd_files):
        filepath = os.path.join(input_folder, filename)
        base, _ = os.path.splitext(filename)
        parts = base.split("-")
        if len(parts) < 3:
            print(f"⚠️ Skipping {filename}: unexpected naming format.")
            continue

        subject = parts[0]
        timepoint = parts[1]
        modality_raw = "-".join(parts[2:]).upper()

        # -----------------------------
        # SUBJECT INDEX PARSING
        # -----------------------------
        try:
            decimal_part = subject.split(".")[1]
            subject_digit = decimal_part[-1]
            if not subject_digit.isdigit():
                raise ValueError("subject digit not numeric")
        except Exception:
            print(f"⚠️ Could not parse subject number from '{subject}' in {filename}. Skipping.")
            continue

        # -----------------------------
        # TIMEPOINT (T0, T1, ...)
        # -----------------------------
        if not (len(timepoint) >= 2 and timepoint[0] == "T" and timepoint[1].isdigit()):
            print(f"⚠️ Skipping {filename}: timepoint '{timepoint}' not recognized.")
            continue

        week_num = int(timepoint[1])
        day_suffix = timepoint_map.get(timepoint, "XXJ")

        # -----------------------------
        # MODALITY DETECTION
        # -----------------------------
        if "SEG" in modality_raw:
            modality = "Mask"
            suffix = "_mask.png"

        elif "WATER" in modality_raw:
            modality = "Water"
            suffix = "_w.png"

        elif "FAT" in modality_raw:
            modality = "Fat"
            suffix = "_f.png"

        else:
            print(f"⚠️ Skipping {filename}: unrecognized modality ({modality_raw}).")
            continue

        # -----------------------------
        # LOAD NRRD
        # -----------------------------
        try:
            data, header = nrrd.read(filepath)
        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")
            continue

        num_slices = data.shape[-1] if data.ndim == 3 else 1
        print(f"📂 {filename}: {num_slices} slices detected")

        # -----------------------------
        # OUTPUT FOLDER
        # -----------------------------
        output_folder = os.path.join(output_base, subject, f"{timepoint}-{day_suffix}", modality)
        os.makedirs(output_folder, exist_ok=True)

        prefix = f"{subject_digit}{week_num}_"

        # -----------------------------
        # SLICE LOOP
        # -----------------------------
        for i in range(num_slices):
            slice_img = np.take(data, i, axis=-1)

            if modality == "Mask":
                # binary mask → 0/255
                slice_img = (slice_img > 0).astype(np.uint8) * 255
                # print(f"slice_img before remove_small_blobs : {np.unique(slice_img)}")
                slice_img = remove_small_blobs(mask=slice_img, plot=False, min_area_fraction=0.0015)
                # print(f"slice_img after remove_small_blobs : {np.unique(slice_img)}")

            

            else:
                # WATER and FAT → normalize per slice and convert to RGB
                slice_img = slice_img.astype(np.float32)
                min_val, max_val = slice_img.min(), slice_img.max()

                if max_val > min_val:
                    slice_img = (slice_img - min_val) / (max_val - min_val)
                else:
                    slice_img = np.zeros_like(slice_img)

                slice_img = (slice_img * 255).astype(np.uint8)
                slice_img = np.stack([slice_img]*3, axis=-1)

            # Rotate to final orientation
            slice_img = np.rot90(slice_img, k=-1)

            out_name = f"{prefix}{i:03d}{suffix}"
            save_path = os.path.join(output_folder, out_name)
            imageio.imwrite(save_path, slice_img)

        rel_output = os.path.join(subject, f"{timepoint}-{day_suffix}", modality)
        print(f"✅ Saved {num_slices} slices for {filename} → {rel_output}\n")

        if modality in ["Water", "Fat"]:
            print("---------------------------------------------------------------------------------------------------")

def split_train_test_df(ds,
                        subject=None,
                        week=None):
    """
    Split a dataset DataFrame into test and train/validation sets based on 
    specific subject(s) and/or week(s).

    Parameters
    ----------
    ds : pd.DataFrame
        Full dataset containing columns ['image_path', 'mask_path', 'subject', 'week'].
    subject : str, float, list, or None
        Subject(s) to include in the test set.
    week : str, int, list, or None
        Week(s) to include in the test set.

    Returns
    -------
    train_val_df : pd.DataFrame
        DataFrame containing all samples NOT in the specified subject/week.
    test_df : pd.DataFrame
        DataFrame containing only the specified subject/week samples.
    """

    df_filtered = ds.copy()

    # Convert subject and week to lists if they are single values
    if subject is not None and not isinstance(subject, (list, tuple, set)):
        subject = [subject]
    if week is not None and not isinstance(week, (list, tuple, set)):
        week = [week]

    # Create boolean mask for test samples (ensure indices align)
    mask = pd.Series(True, index=df_filtered.index)

    if subject is not None:
        mask &= df_filtered["subject"].astype(str).isin([str(s) for s in subject])
    if week is not None:
        mask &= df_filtered["week"].astype(str).isin([str(w) for w in week])

    # Split into test and train_val
    test_df = df_filtered[mask].reset_index(drop=True)
    train_val_df = df_filtered[~mask].reset_index(drop=True)

    train_val_split = (len(train_val_df)/(len(test_df) + len(train_val_df)))*100
    test_split = 100 - train_val_split

    print(f"✅ Test samples: {len(test_df)}, Train/Val samples: {len(train_val_df)}")
    print(f"✅ Train/Test Split : {train_val_split:.2f}/{test_split:.2f}")
    
    return train_val_df, test_df
