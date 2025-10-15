

# Robotics Semester Project II - Spring 2024 : Automatic Segmentation of Light-Sheet Zebrafish Scans

## Project Overview

This repository contains code and files for implementing the finetuning of the newly released vision transformer 'Segment Anything Model' (SAM) onto our custom dataset consisting of light-sheet zebrafish embryo scans at different developmental stages (input to the model), as well as their respective segmentations (expected output).

## Contributors

- Aly Elbindary

## Supervising Professor

- Selman Sakar

## Supervising TA 

- Artur Krzysztof Banach

## Files and Structure

The structure of the repository is the following:

```
ZebrafishSegmentation/
├── data/
│ ├── zebrafish_1/
│ ├── zebrafish_2/
│ ├── zebrafish_3/
│ └── zebrafish_4/
├── multi_data/
│ ├── zebrafish_1/
│ ├── zebrafish_2/
│ ├── zebrafish_3/
│ └── zebrafish_4/
├── raw_data/
│ ├── zebrafish_1/
│ │ ├── Segmentation_1.seg.nrrd
│ │ └── t0001_channel 3.tif
│ ├── zebrafish_2/
│ │ ├── Segmentation_10.seg.nrrd
│ │ └── t0010_channel 3.tif
│ ├── zebrafish_3/
│ │ ├── Segmentation_20.seg.nrrd
│ │ └── t0020_channel 3.tif
│ └── zebrafish_4/
│ │ ├── Segmentation_30.seg.nrrd
│ │ └── t0030_channel 3.tif
├── models/
│ └── base_config_10soms30.pth
│ └── medsam_10soms_epoch10.pth
│ └── model_multi_final_epoch30.pth
├── Aly_Final Presentation.pptx
├── generate_data_folder.py
├── generate_data_singleMask.py
├── README.md
├── requirements.txt
├── tiff_to_jpgs.py
├── train_test_multiSAM.ipynb
└── train_test_SAM.ipynb
```

The most important files and folders are the following:
1. `train_test_multiSAM.ipynb`: This Jupyter Notebook allows for training, testing as well as testing with visualization of our implementation of the multi-class segmentation case of SAM. This current version does not yield intuitive results, and thus further improvements are still needed.

2. `train_test_SAM.ipynb`: This Jupyter Notebook allows for training, testing as well as testing with visualization of our implementation of the binary segmentation case of SAM. This current version yields good results, with features such as the somites and the notochord being clearly learned. Further improvements can still be done, please refer to the project report for further information.

3. `generate_data_folder.py`: This Python script contains functions that convert the data that is within the `raw_data` folder and adapts it into the proper format needed for the multi-class segmentation case.

4. `generate_data_singleMask.py`: This Python script contains functions that convert the data that is within the `raw_data` folder and adapts it into the proper format needed for the binary segmentation case.

## Data

The project data can be found within three different folders : 
- `raw_data` : contains the raw light-sheet scans and manual segmentations generated through the 3D Slicer software ;
- `data` : contains the adapted data for the binary segmentation case ;
- `multi_data` : contains the adapted data for the multi-class segmentation case.

When cloning the repo, make sure to use `git-lfs` in order to properly load the files! More information about `git-lfs` can be found [here](https://github.com/epfl-nlp/cs-552-modern-nlp/blob/main/Exercises/tutorials.md).

## Report & Presentation

The `Semester Project II Report.pdf` file contains the report that has been done on this project. It provides the background highlighting the research that has been done in order to implement this repository, and allows for a better understanding of the context of this project.

The `Aly_Final Presentation.pptx` file contains the slides that have been used for the final presentation.

## Models

The `models` folder contains different checkpoints that we have saved for some of the models that we have trained, that way you can directly test one of the already finetuned SAM models during the project. Currently there are three different model paths :
- `base_config_10soms30.pth` : base SAM model trained on 30 epochs ;
- `medsam_10soms_epoch10.pth` : medsam model trained on 10 epochs ;
- `model_multi_final_epoch30.pth` : multi-class model (using the base SAM model) on 30 epochs.

The original versions of these models can be found on the [HuggingFace](https://huggingface.co/models?other=sam) platform.


## Running the Code

To run the code in this project, follow these steps:

1. Make sure you have the necessary libraries (numpy, and matplotlib), installed in your Python environment. You can set up a Conda environment with the required libraries using the following steps:

```bash
conda create --name zebraSAM python=3.9
conda activate zebraSAM
pip install -r requirements.txt
```

2. In order to utilize your local GPU, CUDA support must be installed by the user if they have a CUDA-capable system. Please refer to the CUDA [documentation](https://pytorch.org/get-started/locally/) and follow the instructions carefully. If the CUDA toolkit is already available, then the following command should enable local GPU usage (be sure to refer to the CUDA documentation if a different command is necessary) :

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Once all the installations are done, then the `CFG.DEVICE` variable within the "Configurations" section of the notebooks should be equal to "`cuda`", otherwise, this would mean that the installation has not been fully completed yet.

Virtual GPU usage could also be done with Google Colab.

3. Run the Notebooks! It is advised to run each cell individually rather than the entire notebook at once, to make sure that everything is in order. Additionally, be careful of certain cells that may take more time than others to execute (training and testing cells namely) as well as certain celss that may not be necessary depending on the context. For usage on Google Colab, certain cells must be used in order to download the appropriate environement requirements as well as mounting the repository onto google drive.

4. ⚠️ <span style="color:red;">ATTENTION!</span> Please note that the cells under the "Adapt Raw Zebrafish Data" section of the notebooks SHOULD NOT be excuted if the data within the  `data` and `multi_data` folders is already there! ⚠️

## Cloning The Repository Without Large Files

All of the data folders as well as the `models` folder have been pushed using `git-lfs`. If the user wishes to clone the repository without the large files that have been pushed using `git-lfs`, then the following commands can be used : 

```bash
$env:GIT_LFS_SKIP_SMUDGE = "1"
git clone https://github.com/alyelbindary/ZebrafishSegmentation.git
```