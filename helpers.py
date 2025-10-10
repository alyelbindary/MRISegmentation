import matplotlib.pyplot as plt
from ipywidgets import interact
import numpy as np
import SimpleITK as sitk
import cv2
import plotly.io as pio
import plotly.graph_objects as go
from skimage import measure
import ants
import pandas as pd
import os

def explore_3D_array(arr: np.ndarray, cmap: str = 'gray'):
  """
  Given a 3D array with shape (Z,X,Y) This function will create an interactive
  widget to check out all the 2D arrays with shape (X,Y) inside the 3D array. 
  The purpose of this function to visual inspect the 2D arrays in the image. 

  Args:
    arr : 3D array with shape (Z,X,Y) that represents the volume of a MRI image
    cmap : Which color map use to plot the slices in matplotlib.pyplot
  """

  arr = arr.transpose(2,1,0)

  def fn(SLICE):
    plt.figure(figsize=(7,7))
    plt.imshow(arr[SLICE, :, :], cmap=cmap)
    plt.show()

  interact(fn, SLICE=(0, arr.shape[0]-1))


def explore_3D_array_comparison(arr_before: np.ndarray, arr_after: np.ndarray, cmap: str = 'gray'):
  """
  Given two 3D arrays with shape (Z,X,Y) This function will create an interactive
  widget to check out all the 2D arrays with shape (X,Y) inside the 3D arrays.
  The purpose of this function to visual compare the 2D arrays after some transformation. 

  Args:
    arr_before : 3D array with shape (Z,X,Y) that represents the volume of a MRI image, before any transform
    arr_after : 3D array with shape (Z,X,Y) that represents the volume of a MRI image, after some transform    
    cmap : Which color map use to plot the slices in matplotlib.pyplot
  """

  assert arr_after.shape == arr_before.shape

  arr_after = arr_after.transpose(2,1,0)
  arr_before = arr_before.transpose(2,1,0)

  def fn(SLICE):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(10,10))

    ax1.set_title('Before', fontsize=15)
    ax1.imshow(arr_before[SLICE, :, :], cmap=cmap)

    ax2.set_title('After', fontsize=15)
    ax2.imshow(arr_after[SLICE, :, :], cmap=cmap)

    plt.tight_layout()
    plt.show()
  
  interact(fn, SLICE=(0, arr_before.shape[0]-1))


def show_sitk_img_info(img: sitk.Image):
  """
  Given a sitk.Image instance prints the information about the MRI image contained.

  Args:
    img : instance of the sitk.Image to check out
  """
  pixel_type = img.GetPixelIDTypeAsString()
  origin = img.GetOrigin()
  dimensions = img.GetSize()
  spacing = img.GetSpacing()
  direction = img.GetDirection()

  info = {'Pixel Type' : pixel_type, 'Dimensions': dimensions, 'Spacing': spacing, 'Origin': origin,  'Direction' : direction}
  for k,v in info.items():
    print(f' {k} : {v}')


def add_suffix_to_filename(filename: str, suffix:str) -> str:
  """
  Takes a NIfTI filename and appends a suffix.

  Args:
      filename : NIfTI filename
      suffix : suffix to append

  Returns:
      str : filename after append the suffix
  """
  if filename.endswith('.nii'):
      result = filename.replace('.nii', f'_{suffix}.nii')
      return result
  elif filename.endswith('.nii.gz'):
      result = filename.replace('.nii.gz', f'_{suffix}.nii.gz')
      return result
  else:
      raise RuntimeError('filename with unknown extension')


def rescale_linear(array: np.ndarray, new_min: int, new_max: int):
  """Rescale an array linearly."""
  minimum, maximum = np.min(array), np.max(array)
  m = (new_max - new_min) / (maximum - minimum)
  b = new_min - m * minimum
  return m * array + b


def explore_3D_array_with_mask_contour(arr: np.ndarray, mask: np.ndarray, thickness: int = 1):
  """
  Given a 3D array with shape (Z,X,Y) This function will create an interactive
  widget to check out all the 2D arrays with shape (X,Y) inside the 3D array. The binary
  mask provided will be used to overlay contours of the region of interest over the 
  array. The purpose of this function is to visual inspect the region delimited by the mask.

  Args:
    arr : 3D array with shape (Z,X,Y) that represents the volume of a MRI image
    mask : binary mask to obtain the region of interest
  """
  assert arr.shape == mask.shape
  
  _arr = rescale_linear(arr,0,1)
  _mask = rescale_linear(mask,0,1)
  _mask = _mask.astype(np.uint8)

  def fn(SLICE):
    arr_rgb = cv2.cvtColor(_arr[SLICE, :, :], cv2.COLOR_GRAY2RGB)
    contours, _ = cv2.findContours(_mask[SLICE, :, :], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    arr_with_contours = cv2.drawContours(arr_rgb, contours, -1, (0,1,0), thickness)

    plt.figure(figsize=(7,7))
    plt.imshow(arr_with_contours)
    plt.show()

  interact(fn, SLICE=(0, arr.shape[0]-1))

  import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact

def explore_3D_blobs(img: np.ndarray, blobs: np.ndarray, cmap: str = 'gray', display_coord = False):
    """
    Visualize 3D volume slices with blob overlays and show blob coordinates + radii.

    Args:
        img : 3D numpy array of shape (X, Y, Z) representing the volume.
        blobs : nx4 numpy array with [x, y, z, r] for each blob.
        cmap : colormap for matplotlib (default: 'gray').
    """

    if display_coord :
       for blob in blobs :
          print(f"Blob coord : ({blob['centroid'][0]}, {blob['centroid'][1]}, {blob['centroid'][2]})")

    def plot_slice(slice_idx: int):
        plt.figure(figsize=(7,7))
        plt.imshow(img[:, :, slice_idx].T, cmap=cmap)
        
        # Overlay blobs in this slice
        for blob in blobs:
            x, y, z, r = blob["centroid"][0], blob["centroid"][1], blob["centroid"][2], blob["radius"]
            if int(round(z)) == slice_idx:  # show only blobs in this slice
                c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
                plt.gca().add_patch(c)
        
        plt.title(f"Slice {slice_idx}")
        plt.axis('off')
        plt.show()

    interact(plot_slice, slice_idx=(0, img.shape[2]-1))

def plot_blobs_in_cube (blobs : np.array, fig_width = 1000, fig_height = 800) :

  # --- Ensure it stays in VSCode ---
  pio.renderers.default = "vscode"

  # --- Convert blobs_props to coordinates (x, y, z only) ---
  coords = [tuple(round(float(v), 2) for v in blob['centroid']) for blob in blobs]
  x = [c[0] for c in coords]
  y = [c[1] for c in coords]
  z = [c[2] for c in coords]

  # --- Cube dimensions ---
  cube_dims = (333, 333, 40)

  # --- Create scatter plot of blobs ---
  scatter = go.Scatter3d(
      x=x,
      y=y,
      z=z,
      mode='markers',
      marker=dict(size=5, color='red'),
      name='Blobs'
  )

  # --- Draw cube edges ---
  cube_lines = []
  vertices = [
      (0,0,0), (cube_dims[0],0,0), (cube_dims[0],cube_dims[1],0), (0,cube_dims[1],0),
      (0,0,cube_dims[2]), (cube_dims[0],0,cube_dims[2]), (cube_dims[0],cube_dims[1],cube_dims[2]), (0,cube_dims[1],cube_dims[2])
  ]
  edges = [
      (0,1),(1,2),(2,3),(3,0),  # bottom face
      (4,5),(5,6),(6,7),(7,4),  # top face
      (0,4),(1,5),(2,6),(3,7)   # vertical edges
  ]
  for e in edges:
      x_line = [vertices[e[0]][0], vertices[e[1]][0], None]
      y_line = [vertices[e[0]][1], vertices[e[1]][1], None]
      z_line = [vertices[e[0]][2], vertices[e[1]][2], None]
      cube_lines.append(go.Scatter3d(
          x=x_line, y=y_line, z=z_line,
          mode='lines',
          line=dict(color='black', width=2),
          showlegend=False
      ))

  # --- Combine scatter and cube ---
  fig = go.Figure(data=[scatter] + cube_lines)

  # --- Exaggerate z-axis for visualization ---
  z_scale_factor = 7.5  # try values between 3–10 for more visual balance

  aspect_ratio = dict(
      x=1,
      y=1,
      z=(cube_dims[2] / cube_dims[0]) * z_scale_factor
  )

  fig.update_layout(
      scene=dict(
          xaxis=dict(range=[0, cube_dims[0]]),
          yaxis=dict(range=[0, cube_dims[1]]),
          zaxis=dict(range=[0, cube_dims[2]]),
          aspectmode='manual',
          aspectratio=aspect_ratio
      ),
      title="3D Blob Coordinates within an MRI Volume",
      width=fig_width,
      height=fig_height
  )

  fig.show()

def filter_mask_by_cube (
        mask: sitk.Image,
        x_range=[0, 333],
        y_range=[0, 333],
        z_range=[0, 40],
        value_to_keep=None,
        output_type=sitk.sitkUInt32):
    """
    Keep only voxels in `mask` within specified index ranges along x, y, z axes.
    
    Parameters
    ----------
    mask : sitk.Image
        Input mask image (SimpleITK instance)
    x_range : list or tuple of two ints [xmin, xmax]
    y_range : list or tuple of two ints [ymin, ymax]
    z_range : list or tuple of two ints [zmin, zmax]
    value_to_keep : int, optional
        Only keep voxels equal to this value. If None, all nonzero voxels are considered.
    output_type : SimpleITK pixel type
        Pixel type of output mask
    
    Returns
    -------
    sitk.Image
        Filtered mask
    """
    # Convert to NumPy array
    mask_arr = sitk.GetArrayFromImage(mask)  # shape: (z, y, x)
    z_dim, y_dim, x_dim = mask_arr.shape

    # Build coordinate grids matching (z, y, x)
    z_grid, y_grid, x_grid = np.meshgrid(
        np.arange(z_dim), np.arange(y_dim), np.arange(x_dim), indexing='ij'
    )

    # Build condition
    condition = (
        (x_grid >= x_range[0]) & (x_grid <= x_range[1]) &
        (y_grid >= y_range[0]) & (y_grid <= y_range[1]) &
        (z_grid >= z_range[0]) & (z_grid <= z_range[1])
    )

    if value_to_keep is not None:
        condition &= (mask_arr == value_to_keep)
    else:
        condition &= (mask_arr != 0)

    # Apply condition
    filtered_arr = np.zeros_like(mask_arr, dtype=np.uint32)
    filtered_arr[condition] = 1

    # Convert back to SITK
    filtered_mask = sitk.GetImageFromArray(filtered_arr)
    filtered_mask.CopyInformation(mask)
    
    return sitk.Cast(filtered_mask, output_type)

def zero_mask_by_cube(
        mask: sitk.Image,
        x_range=[0, 333],
        y_range=[0, 333],
        z_range=[0, 40],
        value_to_zero=None,
        output_type=sitk.sitkUInt32):
    """
    Set voxels to zero in `mask` that are within specified index ranges along x, y, z axes.
    
    Parameters
    ----------
    mask : sitk.Image
        Input mask image (SimpleITK instance)
    x_range : list or tuple of two ints [xmin, xmax]
        Zero voxels with x indices in this range (inclusive)
    y_range : list or tuple of two ints [ymin, ymax]
        Zero voxels with y indices in this range (inclusive)
    z_range : list or tuple of two ints [zmin, zmax]
        Zero voxels with z indices in this range (inclusive)
    value_to_zero : int, optional
        Only zero voxels equal to this value. If None, all nonzero voxels are zeroed.
    output_type : SimpleITK pixel type
        Pixel type of output mask
    
    Returns
    -------
    sitk.Image
        Modified mask with voxels inside cube zeroed
    """
    # Convert to NumPy array
    mask_arr = sitk.GetArrayFromImage(mask)  # shape: (z, y, x)
    z_dim, y_dim, x_dim = mask_arr.shape

    # Build coordinate grids matching (z, y, x)
    z_grid, y_grid, x_grid = np.meshgrid(
        np.arange(z_dim), np.arange(y_dim), np.arange(x_dim), indexing='ij'
    )

    # Build condition for voxels inside the cube
    condition = (
        (x_grid >= x_range[0]) & (x_grid <= x_range[1]) &
        (y_grid >= y_range[0]) & (y_grid <= y_range[1]) &
        (z_grid >= z_range[0]) & (z_grid <= z_range[1])
    )

    # Apply value filter if specified
    if value_to_zero is not None:
        condition &= (mask_arr == value_to_zero)
    else:
        condition &= (mask_arr != 0)

    # Zero out voxels inside the cube
    mask_arr[condition] = 0

    # Convert back to SITK image
    filtered_mask = sitk.GetImageFromArray(mask_arr)
    filtered_mask.CopyInformation(mask)
    
    return sitk.Cast(filtered_mask, output_type)

def get_blobs (mask : ants.core.ants_image.ANTsImage, radius_thresh = 6., display = False) -> np.array :
    """
    This function determines the relevant blobs corresponding to the implants within the scans
    we retreive the blobs's centroid corrdinates, the equivalent radius as well as both the 
    major axis and minor axis lengths and the elongation. Only blobs that have an equivalent radius superior to the
    parameter "radius_thresh" are kept, as the are considered to be ones corresponding to the
    implants. The "Region Props" algorithm is utilized to identify the blobs.

    Parameetrs :
        - mask : antspy image correponding to the mask ;
        - radius_threshold : float correponding the threshold radius.
    
    Outputs :
        - blobs numpy array corresponding to the different blob parameters.
    """
     # --- Prepare the mask ---
    mask.set_origin((0,0,0))
    mask_numpy = mask.numpy()

    # --- Connected component analysis ---
    labels = measure.label(mask_numpy)  # each connected region gets a unique integer label
    props = measure.regionprops(labels)

    # Extract blob information: centroid and approximate radius
    blobs = []

    for i, prop in enumerate(props) :
        try:
            centroid = tuple(prop.centroid)
            major = prop.axis_major_length
            minor = prop.axis_minor_length
            principal = 4 * np.sqrt(prop.inertia_tensor_eigvals[-1])
            radius = prop.equivalent_diameter / 2  # example, can be modified
            if minor != 0:
                elongation = major / minor

            if (radius >= radius_thresh) :
                # blobs.append((*centroid, radius, prop.axis_major_length, prop.axis_minor_length, prop.principal_axis_length, elongation))
                if display :
                    print(f"Region {i}: major={major:.3f}, minor={minor:.3f}, elongation={elongation:.3f}, "
                        f"diameter = {prop.equivalent_diameter:.3f}, major axis length = {prop.major_axis_length:.3f}, "
                        f"minor axis length = {prop.minor_axis_length:.3f}, principal axis length = {principal:.3f}")
                    print("--------------------------------------------------------------------------------------" \
                    "-------------------------------------------------------------------------------------------")

                blob = {
                   
                  'centroid': centroid,
                  'radius': radius,
                  'major_axis_length': major,
                  'minor_axis_length': minor,
                  'principal_axis_length': principal,
                  'elongation': elongation
                }

                blobs.append(blob)


        except ValueError as e:
            print(f"⚠️ Region {i} skipped due to math domain error: {e}\n")

    # Sort blobs based on radius (descending)
    blobs = sorted(blobs, key=lambda b: b['radius'], reverse=True)

    if display :
        print(f"Number of Blobs: {len(blobs)}")

    return blobs

def get_volumes(blobs, margin_ratio=0.1):
    """
    Create rectangular bounding volumes around ellipsoidal blobs, 
    expanded by a given margin ratio per axis.

    Parameters
    ----------
    blobs : list of tuples
        Each entry is (x, y, z, radius, major_axis, minor_axis, elongation)
    margin_ratio : float, optional
        Percentage margin to extend beyond each axis (default = 0.1)

    Returns
    -------
    volumes : list of dict
        Each dict contains:
            - 'center': (x, y, z)
            - 'bounds': ((x_min, x_max), (y_min, y_max), (z_min, z_max))
            - 'sizes': (x_length, y_length, z_length)
    """

    # --- Return empty list if no blobs ---
    if not blobs:
        return []

    volumes = []

    for blob in blobs:
        cx, cy, cz = blob['centroid']  # x, y, z
        major_axis = blob['major_axis_length']      # → x
        principal_axis = blob['principal_axis_length']  # → y
        minor_axis = blob['minor_axis_length']      # → z

        # Half-lengths with margin
        half_x = (major_axis / 2) * (1 + margin_ratio)
        half_y = (principal_axis / 2) * (1 + margin_ratio)
        half_z = (minor_axis / 2) * (1 + margin_ratio)

        # Cuboid bounds
        x_min, x_max = cx - half_x, cx + half_x
        y_min, y_max = cy - half_y, cy + half_y
        z_min, z_max = cz - half_z, cz + half_z

        volumes.append({
            "center": (cx, cy, cz),
            "bounds": ((x_min, x_max), (y_min, y_max), (z_min, z_max)),
            "sizes": (2 * half_x, 2 * half_y, 2 * half_z)
        })

    return volumes

def plot_blobs_and_volumes(blobs, volumes, cube_dims=(333, 333, 40), z_scale_factor=7.5, fig_width = 1000, fig_height = 800):
    """
    Plot 3D ellipsoids (blobs) and their rectangular bounding volumes inside a cube.

    Parameters
    ----------
    blobs : list of dicts
        Each blob dict must contain:
            - 'centroid': (x, y, z)
            - 'major_axis_length'
            - 'minor_axis_length'
            - 'principal_axis_length'
    volumes : list of dicts
        Each dict must contain:
            - 'bounds': ((x_min, x_max), (y_min, y_max), (z_min, z_max))
    cube_dims : tuple
        Dimensions of the cube (default = (333, 333, 40))
    z_scale_factor : float
        Factor to exaggerate the z-axis for visualization
    """

    pio.renderers.default = "vscode"

    # --- Draw outer cube edges ---
    cube_lines = []
    vertices = [
        (0,0,0), (cube_dims[0],0,0), (cube_dims[0],cube_dims[1],0), (0,cube_dims[1],0),
        (0,0,cube_dims[2]), (cube_dims[0],0,cube_dims[2]), (cube_dims[0],cube_dims[1],cube_dims[2]), (0,cube_dims[1],cube_dims[2])
    ]
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]
    for e in edges:
        x_line = [vertices[e[0]][0], vertices[e[1]][0], None]
        y_line = [vertices[e[0]][1], vertices[e[1]][1], None]
        z_line = [vertices[e[0]][2], vertices[e[1]][2], None]
        cube_lines.append(go.Scatter3d(
            x=x_line, y=y_line, z=z_line,
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False
        ))

    # --- Plot ellipsoids ---
    ellipsoids = []
    for i, blob in enumerate(blobs):
        xc, yc, zc = blob['centroid']
        major_axis = blob['major_axis_length']
        minor_axis = blob['minor_axis_length']
        principal_axis = blob['principal_axis_length']

        # Meshgrid for ellipsoid
        u = np.linspace(0, 2 * np.pi, 40)
        v = np.linspace(0, np.pi, 20)
        u, v = np.meshgrid(u, v)

        x = (major_axis / 2) * np.cos(u) * np.sin(v) + xc       # major → x
        y = (principal_axis / 2) * np.sin(u) * np.sin(v) + yc   # principal → y
        z = (minor_axis / 2) * np.cos(v) + zc                   # minor → z

        ellipsoids.append(go.Surface(
            x=x, y=y, z=z,
            opacity=0.4,
            colorscale=[[0, 'red'], [1, 'red']],
            showscale=False,
            name=f"Ellipsoid {i+1}"
        ))

    # --- Plot rectangular bounding boxes ---
    box_lines = []
    for i, v in enumerate(volumes):
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = v['bounds']
        verts = [
            (x_min, y_min, z_min), (x_max, y_min, z_min),
            (x_max, y_max, z_min), (x_min, y_max, z_min),
            (x_min, y_min, z_max), (x_max, y_min, z_max),
            (x_max, y_max, z_max), (x_min, y_max, z_max)
        ]
        edges = [
            (0,1),(1,2),(2,3),(3,0),
            (4,5),(5,6),(6,7),(7,4),
            (0,4),(1,5),(2,6),(3,7)
        ]
        for e in edges:
            x_line = [verts[e[0]][0], verts[e[1]][0], None]
            y_line = [verts[e[0]][1], verts[e[1]][1], None]
            z_line = [verts[e[0]][2], verts[e[1]][2], None]
            box_lines.append(go.Scatter3d(
                x=x_line, y=y_line, z=z_line,
                mode='lines',
                line=dict(color='blue', width=3),
                showlegend=False
            ))

    # --- Combine everything ---
    fig = go.Figure(data=cube_lines + box_lines + ellipsoids)

    # --- Adjust aspect ratio (exaggerate z) ---
    aspect_ratio = dict(
        x=1,
        y=1,
        z=(cube_dims[2]/cube_dims[0]) * z_scale_factor
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[0, cube_dims[0]]),
            yaxis=dict(range=[0, cube_dims[1]]),
            zaxis=dict(range=[0, cube_dims[2]]),
            aspectmode='manual',
            aspectratio=aspect_ratio,
        ),
        title="3D Ellipsoids and Bounding Boxes within Volume",
        width=fig_width,
        height=fig_height
    )

    fig.show()

def create_nrrd_dataframe_with_ants(folder_path):
    """
    Scans a folder for .nrrd files, creates a DataFrame with columns:
    'File', 'Rat', 'Week', 'Scan Type', and loads each as an ants image.
    
    Args:
        folder_path (str): Path to the folder containing .nrrd files.
    
    Returns:
        pd.DataFrame
    """
    records = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".nrrd"):
            parts = filename.split('-')
            if len(parts) >= 3:
                rat = parts[0]
                week = parts[1]
                scan_type = parts[2].replace(".nrrd", "")
                
                # full path to the file
                file_path = os.path.join(folder_path, filename)
                
                # load as ants image
                ants_img = ants.image_read(file_path)
                
                records.append({
                    "File": filename,
                    "Rat": rat,
                    "Week": week,
                    "Scan Type": scan_type,
                    "ANTsImage": ants_img
                })
    
    df = pd.DataFrame(records)
    return df

def create_mask (image, display = False) -> ants.core.ants_image.ANTsImage :
    """
    This function takes in an sitk.image, apply Otsu thresholding and applies some preprocessing techniques
    in order to generate an appropriate corresponding mask. The operation are done in the following order :
    OtsuThresholding => Cropping => Fill Holes => Dilation => Opening => Closing.
    The image variable can also be given as an antspy image, and the proper conversions will be done.

    Parameters :
        - image : sitk.Image corresponding to the MRI scan volume (333 x 333 x 40)
    
    Outputs :
        - figure displaying the before and after results ;
        - mask : antspy image corresponding to the resulting mask.
    """

    if isinstance(image, ants.core.ants_image.ANTsImage) :
        image_antspy = image

        # Convert to NumPy + extract metadata
        arr = image.numpy()
        arr = arr.transpose(2,1,0)
        spacing = image.spacing
        origin = image.origin
        direction = image.direction.flatten()

        # Create a SimpleITK image
        image = sitk.GetImageFromArray(arr)
        image.SetSpacing(spacing)
        image.SetOrigin(origin)
        image.SetDirection(direction)

    else :

        # Get ANTsPy equivalent image from sitk image
        image = sitk.Cast(image, sitk.sitkUInt32)
        image_antspy = ants.from_sitk(image)
        image_antspy = ants.from_sitk(image)

    # Create Otsu mask
    mask_otsu = sitk.OtsuMultipleThresholds(image, numberOfThresholds=4)
    mask_otsu_4_classes = sitk.Cast(mask_otsu, sitk.sitkUInt32)
    mask_otsu_4_classes = ants.from_sitk(mask_otsu_4_classes)

    mask_otsu_4 = sitk.Cast(mask_otsu == 4, sitk.sitkUInt32)

    mask_otsu_3_4 = sitk.Cast((mask_otsu == 4) | (mask_otsu == 3), sitk.sitkUInt32)
    mask_otsu_3_4 = ants.from_sitk(mask_otsu_3_4)

    # Create Copped Mask
    mask_otsu_3_cropped = filter_mask_by_cube (mask_otsu == 3, x_range=[0, 135], y_range=[0,100])

    mask_otsu_3_4_cropped_array = np.logical_or(
        sitk.GetArrayFromImage(mask_otsu_4),
        sitk.GetArrayFromImage(mask_otsu_3_cropped)).astype(np.uint32)

    mask_3_4_cropped = sitk.GetImageFromArray(mask_otsu_3_4_cropped_array)
    mask_3_4_cropped.SetOrigin(mask_otsu_4.GetOrigin())
    mask_3_4_cropped.SetSpacing(mask_otsu_4.GetSpacing())
    mask_3_4_cropped.SetDirection(mask_otsu_4.GetDirection())

    mask_ultra_cropped = zero_mask_by_cube(mask_3_4_cropped, y_range = [100, 150], z_range=[18, 40])

    mask_ultra_cropped.SetOrigin(mask_otsu_4.GetOrigin())
    mask_ultra_cropped.SetSpacing(mask_otsu_4.GetSpacing())
    mask_ultra_cropped.SetDirection(mask_otsu_4.GetDirection())

    mask_otsu_binary = sitk.Cast(mask_ultra_cropped, sitk.sitkUInt32)

    # Convert to ANTsPy image
    mask_otsu_binary = ants.from_sitk(mask_otsu_binary)

    # Fill Holes
    mask_fillholes = ants.iMath(image=mask_otsu_binary.astype("float32"), operation="FillHoles")

    # Dilation
    radius_dilation = 0.9
    mask_dilat = ants.morphology(
        image=mask_otsu_binary.astype("float32"),
        operation="dilate",
        mtype="binary",
        radius=radius_dilation
    )

    # Opening
    radius_opening = 1.
    mask_dilat_opened = ants.morphology (
        mask_dilat, 
        radius=radius_opening,
        operation="open",
        mtype = "grayscale"
    )

    # Closing
    radius_closing = 3.
    mask_dilat_open_closed = ants.morphology (
        mask_dilat_opened, 
        radius=radius_closing,
        operation="close",
        mtype = "grayscale"
    )

    # Show before and after results if display parameter is equal to true
    if (display) :
        explore_3D_array_comparison(
            arr_before=image_antspy.numpy(),
            arr_after=mask_dilat_open_closed.numpy()
        )
    
    return mask_dilat_open_closed
