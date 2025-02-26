<span style="color:red;">**Go through the file and fix what is necessary or create a new with similar idea.**</span>
## Overview

This project consists of a collection of Python scripts and modules aimed at processing, analyzing, and visualizing image stack data, with extensive support for video data, displacement calculation, and visualization. This README provides an overview of the codebase and the major functionalities it supports.

### Key Functionalities
1. **Image Registration and Processing**: Aligns images, calculates displacements, and applies transformations.
2. **Visualization Tools**: Generates plots for visualizing displacement, image slices, and fitted ellipses or circles.
3. **Video Creation**: Converts processed image stacks into annotated videos displaying calculated transformations.
4. **Correlation Analysis**: Calculates and visualizes correlations across video frames or image stacks.
5. **Utility Functions**: Supports a wide range of matrix processing operations, including masking, variance calculations, and frame rotation.

---

### Folder Contents

Below is a description of the individual scripts and their main responsibilities within the project.

---

### **1. `register_matrix.py`**

#### Purpose:
Defines core functionality for image registration and transformation.

#### Key Functionalities:
- **Image Alignment**:
  - `apply_displacement`: Applies calculated displacement to align an image stack.
  - `register_image_stack`: Registers a stack of images through displacement correction.
- **Image Transformations**:
  - `shift_by_vector`: Shifts images using a specified displacement vector.
  - `compute_affine_transform`: Computes affine transformations for geometric alignment.
  - `apply_transformations`: Applies transformations such as rotation, scaling, or translation.
- **Data Extraction**:
  - `extract_means` and `extract_medians`: Extract statistical data such as means or medians from image stacks.
  - `draw_means`: Visualizes the calculated means of the image stack.

---

### **2. `create_video_from_stack.py`**

#### Purpose:
Converts an image stack into an MP4 video while highlighting displacement data.

#### Key Functionalities:
- **Input**: Accepts `.npy` image stacks and associated displacement data stored in `_displacement.npz` format.
- **Processing**:
  - Uses displacement data to display annotation points (e.g., green circles for mesh nodes, center annotation).
- **Video Creation**:
  - Compiles the annotated frames into a video.
  - Saves the final video with MP4 format using the `cv.VideoWriter` functionality.

---

### **3. `draw_displacement.py`**

#### Purpose:
Visualizes displacement data on a 2D plot.

#### Key Functionalities:
- **Displacement Visualization**:
  - `draw_displacement`: Generates color-coded plots of displacement data.
- **Parameters for Configuration**:
  - Enhances customization of input parameters for plotting via the CLI.

---

### **4. `slice_image_stack.py`**

#### Purpose:
Partitions an image stack into slices for easier correlation analysis.

#### Key Functionalities:
- **Image Slicing**:
  - Splits an image stack at fixed intervals into smaller groups (`gallery` of frames).
- **Visualization**:
  - Uses external visualizers to display or save the processed slices.

---

### **5. `mudic_correlation.py`**

#### Purpose:
Performs correlation analysis between images or across multiple frames of a video.

#### Key Functionalities:
- **Mesh Creation and Correlation**:
  - `create_mesh`: Constructs a computational mesh for analyzing image data.
  - `correlate_matrix`: Computes correlation between elements of the matrix.
- **Mesh Nodes Extraction**:
  - Extracts mesh nodes required for advanced displacement algorithms.

---

### **6. `find_circle.py`**

#### Purpose:
Fits a circle or an ellipse to displacement data for detecting rotational patterns or centers.

#### Key Functionalities:
- **Fitting Shapes**:
  - `find_circle_for_center`: Uses displacement data to fit a circle.
  - `find_ellipse`: Fits an ellipse and plots it.
- **Visualization**:
  - Provides plots to visually confirm the fitted circles or ellipses.

---

### **7. `matrix_processing.py`**

#### Purpose:
Handles preprocessing and analysis of image or video matrix data.

#### Key Functionalities:
- **Matrix Representations**:
  - Implements functions such as `min_image`, `max_image`, and `variance_image` to extract statistical representations of images.
- **Masking**:
  - Employs `mask_img_with_min` for masking irrelevant areas in images.
- **Histogram Generation**:
  - Generates and saves histograms for visual analysis.

---

### **8. `video_matrix.py`**

#### Purpose:
Provides helper functions for video-related operations.

#### Key Functionalities:
- **Video Frame Processing**:
  - `rotate_frames`: Processes frames to align or adjust them.
  - `create_video_matrix`: Converts video into a processable format (e.g., matrix representation).
- **Progress Tracking**:
  - Implements `tqdm_generator` for progress visualization during operations.

---

### Common Utilities and Structure
The scripts are designed with modularity in mind. Some re-usable modules, such as `matrix_processing` or `visualisers`, ensure consistent and efficient processing pipelines for correlated tasks.

---

### How to Use

1. **Installation**:
   - Place all scripts in the same directory.
   - Make sure to install the required dependencies:
```shell script
pip install numpy opencv-python matplotlib scikit-image tqdm
```

2. **Execution**:
   - Each script has a `main` function that can be executed independently via the CLI.
   - Use the `--help` flag for any script to see its available options.

3. **Example Workflow**:
   - Start by processing the video/image stack using `matrix_processing.py`.
   - Use `register_matrix.py` and `mudic_correlation.py` to perform registration and calculate displacement.
   - Plot and analyze displacements using `draw_displacement.py`.
   - Finally, compile results into a video with `create_video_from_stack.py`.

---

### Dependencies

The following Python libraries are required for this project:
- **Numerical Computation**: `numpy`, `scipy`
- **Visualization**: `matplotlib`
- **Video Processing**: `opencv-python`
- **File Handling**: `argparse`, `os`
- **Progress Feedback**: `tqdm`
- **Image Matching**: `scikit-image`

---

### TODOs and Future Work

- Add logging across all scripts for better debug output and monitoring during long processes.
- Standardize the input/output formats for improved interoperability.
- Enhance support for configurable parameters, moving away from hardcoded values.
- Improve documentation for generated files (e.g., `_displacement.npz`, videos).
- Create a unified command-line tool or Python package for easier access to all functionalities.

---

This documentation provides a general overview and workflow for using the folder’s functionality. For further details or updates, please refer to the scripts directly.