from wand.image import Image as WandImage
import os
import nrrd
import numpy as np
from PIL import Image as PilImage

"""
Rotate .nrrd data in different ways
"""
def rotate_around_middle_horizontal_axis(data, angle):
    # Get the height and width of each frame
    height, width, num_frames = data.shape
    
    # Calculate the middle index along the horizontal axis for each frame
    middle_horizontal_indices = width // 2
    
    # Initialize an array to store the rotated data
    rotated_data = np.empty_like(data)
    
    # Iterate over each frame
    for i in range(num_frames):
        # Rotate the data within the frame around the middle horizontal axis
        rotated_frame = np.rot90(data[:, :, i], k=angle // 90, axes=(1, 0))
        
        # Flip the rotated frame along the horizontal axis to align with the original orientation
        rotated_frame = np.flip(rotated_frame, axis=1)
        
        # Store the rotated frame in the output array
        rotated_data[:, :, i] = rotated_frame
    
    return rotated_data

def rotate_data(data, angle):
    # Rotate the data by the specified angle
    rotated_data = np.rot90(data, k=angle//90, axes=(0, 1))  # Adjust axes if needed
    return rotated_data

"""
TIF input files
"""
def split_multi_page_tiff(input_file, output_folder, file_prefix):
    """
    This function takes in a zebrafish.tif file, which contains hundreds of frames,
    and generates an individual .tif file for each frame (in order to match 
    the training pipeline data structure). Saves them in a folder

    Inputs : 
        - input_file : zebrafish tif file path
        - output_folder : folder path where the files will be saved
        - file_prefix : string representing the prefix with which we're naming the
                        files


    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with WandImage(filename=input_file) as img:
        for i, page in enumerate(img.sequence):
            with WandImage(page) as single_img:
                output_path = os.path.join(output_folder, f"{file_prefix}_frame_{i}.tif")
                single_img.save(filename=output_path)
                single_img.clear()


"""
TIF Mask Labels
"""
def convert_nrrd_to_tiffs(input_file_path, output_folder, file_prefix):
    """
    This function takes in a segmentation.nrrd file, which contains hundreds of frames,
    (corresponding to segmentation masks) and generates an individual tif file for each
    mask (in order to match the training pipeline data structure). Saves them in a folder

    Inputs : 
        - input_file : segmentation.nrrd file path
        - output_folder : folder path where the files will be saved
        - file_prefix : string representing the prefix with which we're naming the
                        files


    """
    # Get data in .nrrd file
    data, _ = nrrd.read(input_file_path)

    # PROPER ROTATION CONFIGURATION 
    if len(data.shape) == 3 :
        data = rotate_data(data, 90)
        data = rotate_around_middle_horizontal_axis(data, 180)
    elif len(data.shape) == 4 :
        data = data[0, :, :, :]  + data[1, :, :, :]
        data = rotate_data(data, 90)
        data = rotate_around_middle_horizontal_axis(data, 180)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    num_frames = data.shape[2]
    for i in range(num_frames):
        frame_data = data[:, :, i]
        
        # Normalize data to 0-255 range
        normalized_frame = ((frame_data - np.min(frame_data)) / (np.max(frame_data) - np.min(frame_data)) * 255).astype(np.uint8)
        
        # Convert to PIL Image
        frame_image = PilImage.fromarray(normalized_frame)
        
        # Save as TIFF file
        frame_image.save(os.path.join(output_folder, f"{file_prefix}_frame_{i}_mask.tif"))


"""
MEGA FUNCTION FOR DATA
"""
def generate_data_folder (tif_file_path, nrrd_file_path, zebrafish_number) :
    folder = f"data/zebrafish_{zebrafish_number}"
    file_prefix = f"zebrafish_{zebrafish_number}"

    # Generate Input frames
    split_multi_page_tiff(tif_file_path, folder, file_prefix)

    # Generate Corresponding Masks
    convert_nrrd_to_tiffs(nrrd_file_path, folder, file_prefix)