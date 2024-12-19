# Traffic-Vehicle-Detection-BKAI

This repository contains a traffic vehicle detection system developed using deep learning models and image processing techniques. Below, you will find details about the dataset preparation, model training, and evaluation process.

## Dataset Preparation

### Split the Dataset

The dataset is split into training and validation sets using the following command:

```bash
python split_dataset_tvt.py --train 90 --validation 10 --folder root_data/train --dest data/train_dataset
```

### Count Images

The following script counts the number of images in the train and validation directories:

```python
import os

# Paths to train and validation directories
train_images_path = './data/train_dataset/images/train'
val_images_path = './data/train_dataset/images/val'

# Function to count images
def count_images(path):
    return len([file for file in os.listdir(path) if file.endswith(('.jpg', '.png'))])

# Print the counts
print("Number of training images:", count_images(train_images_path))
print("Number of validation images:", count_images(val_images_path))
```

## Image Processing

### Dehazing and Enhancement

Image enhancement techniques like dehazing and ESRGAN were applied to improve the quality of images before training.

## Model Training

### YOLO Models

The training script supports YOLO models. An example of training a YOLO model:

```bash
python train.py --data data.yaml --cfg yolov8.yaml --weights yolov8.pt --epochs 50
```

### Handle Unbalanced Classes

We addressed the issue of unbalanced classes using the `YoloWeightedDataset`, ensuring better representation of minority classes.

## Custom Model Architectures

The architectures were modified to include the following components:

- DySample
- DyHead
- C2f\_DWRSeg
- C3k2\_DWRSeg
- ResBlock
- CBAM

## Tools and Technologies

- [Ultralytics YOLO](https://github.com/ultralytics/yolov8)
- OpenCV

## Achievements

- Improved detection accuracy and robustness in various weather conditions and lighting scenarios.

## How to Run

1. Clone the repository:

    ```bash
    git clone https://github.com/username/TrafficVehicleDetection.git
    cd TrafficVehicleDetection
    ```

2. Prepare the dataset:

    Follow the steps in the "Dataset Preparation" section.

3. Train the model:

    Use the training script provided in the "Model Training" section.

4. Evaluate the model:

    Run the evaluation script:

    ```bash
    python evaluate.py --weights best_model.pt --data data.yaml
    ```

