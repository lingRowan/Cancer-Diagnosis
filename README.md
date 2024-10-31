
# Brain MRI Classification Project

## Overview
This project implements a Convolutional Neural Network (CNN) to classify MRI scans of the brain into different categories. Utilizing PyTorch for building and training the model, the project covers data processing, model architecture, training processes, and evaluation metrics. It aims to assist in the diagnosis of brain conditions by analyzing MRI images.

## Dataset
The project uses a dataset of brain MRI scans organized into separate folders for training and validation:
- **Training Dataset**: Contains MRI images for training the model.
- **Validation Dataset**: Contains MRI images for evaluating the model's performance.

The images are expected to be structured as follows:
```
Brain_MRI_Images/
    ├── Train/
        ├── class_0/
        ├── class_1/
        └── class_2/
    └── Validation/
        ├── class_0/
        ├── class_1/
        └── class_2/
```

## Features
- **Image Preprocessing**: The images are resized to 128x128 pixels and converted into tensor format for input into the CNN.
- **CNN Architecture**: The model consists of multiple convolutional layers followed by max pooling, dropout layers for regularization, and a fully connected layer for producing class predictions.
- **Training and Validation**: The model is trained using training datasets, and its performance is evaluated on validation datasets. The model utilizes the Adam optimizer and cross-entropy loss for training.
- **Metrics Tracking**: The project tracks training loss, validation loss, and accuracy for each batch during the training process.

## Required Libraries
- `torch`
- `torchvision`
- `numpy`
- `matplotlib`

## Installation
To run this project, make sure you have the required libraries installed. You can install them using pip:

```bash
pip install torch torchvision numpy matplotlib
```

## How to Use
1. Set up your dataset as indicated above.
2. Update the paths in the code base to your dataset location.
3. Run the script to train the model and evaluate its performance.

## Example Results
After training, this model can classify brain MRI scans into specified categories. The average loss and accuracy on the validation set can be monitored, ensuring that the model is learning effectively.

### Visualization
The code generates plots for validation loss and accuracy, providing insights into the model’s performance over batches.

```plaintext
Validation set: Average loss: 2.113802, Accuracy: 67/80 (84%)
```

![Per Batch](<Validation_loss_Accuracy_per_batch.png>)
![Accuracy and Loss Plot](<Validation_loss_Accuracy.png>)
