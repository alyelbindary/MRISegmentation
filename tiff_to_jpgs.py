import numpy as np
from PIL import Image

def adjust_jpg(image):
    """
    Takes in a jpeg that has been extracted from a .tif file and 
    adjusts the pixel intensity in order to get a readable image

    Inputs : 
        - image : original jpeg to be adjusted
    Outputs :
        - image : adjusted jpeg
    """
    # Convert image to uint8 type
    if image.dtype != np.uint8:
        image = (image / (np.max(image) / 255)).astype(np.uint8)

    return image

def tiff_to_jpgs(tiff_file, output_path):
    """
    This function takes in the path to the .tif file and geneartes a
    JPEG for each individual frame

    Inputs : 
        - tiff_file : .tif file path
        - output_path : folder path in which we'll save the frames
    """
    # Open the TIFF file
    tiff_img = Image.open(tiff_file)

    # Loop through each frame and save as JPEG
    for i in range(tiff_img.n_frames):
        
        # Go to the i-th page
        tiff_img.seek(i)

        
        # Convert image to 8-bit grayscale mode ('L')
        converted_img = np.array(tiff_img)

        # Convert Imaget to proper format :
        converted_img = Image.fromarray(adjust_jpg(converted_img))
        
        # Save the current page as JPEG
        converted_img.save(f"{output_path}/frame_{i+1}.jpg", format="JPEG")