# LaneNet Postprocessing

LaneNet Postprocessing is a C++ application that performs postprocessing on LaneNet model outputs to obtain binary lane segmentations and instance segmentations. It includes clustering and visualization techniques to enhance the interpretability of the model's predictions.

## Features

- **Binary Segmentation:** Converts LaneNet's pixel-wise predictions into binary lane segmentations.

- **Instance Segmentation:** Performs clustering on pixel embeddings to create instance segmentations of individual lanes.

- **DBSCAN Clustering:** Utilizes the Density-Based Spatial Clustering of Applications with Noise (DBSCAN) algorithm to group pixel embeddings into clusters representing lanes.

- **Visualization:** Provides visualization of binary and instance segmentation results for qualitative analysis.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/patrickeala/lanenet-postprocess.git
   cd lanenet-postprocess
   ```

2. Build the project using CMake:
   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```

## Usage

1. Update the `main.cpp` file with the appropriate input directory containing LaneNet model outputs.

2. Compile and run the program:
   ```bash
   ./lanenet_postprocess
   ```

3. The program will generate binary and instance segmentation images in the specified input directory and display the visualizations.

