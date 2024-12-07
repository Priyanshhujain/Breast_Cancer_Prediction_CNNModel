# Breast Cancer Detection using CNN

This project implements a Convolutional Neural Network (CNN) to classify breast cancer as benign or malignant based on medical imaging data. The model processes MRI images, extracts patterns, and predicts the presence of breast cancer.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
Early detection of breast cancer can save lives. This project leverages deep learning techniques to classify breast cancer efficiently. The model employs CNNs to analyze MRI images and distinguish between benign and malignant tumors.

## Dataset
The dataset used in this project consists of breast MRI images labeled as benign or malignant. Ensure the dataset is structured as follows:

```
Dataset/
|-- Benign/
|   |-- image1.jpg
|   |-- image2.jpg
|   |-- ...
|-- Malignant/
    |-- image1.jpg
    |-- image2.jpg
    |-- ...
```

## Prerequisites
- Python 3.7+
- Jupyter Notebook
- TensorFlow
- NumPy
- Matplotlib
- Pandas

You can install the required packages using the command:
```bash
pip install -r requirements.txt
```

## Installation
1. Clone the repository:
```bash
git clone https://github.com/your-username/breast-cancer-detection-cnn.git
```

2. Navigate to the project directory:
```bash
cd breast-cancer-detection-cnn
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Place your dataset in the `Dataset/` directory as per the structure mentioned above.
2. Open the Jupyter notebook:
```bash
jupyter notebook Breast_Cancer_Detection_using_CNN.ipynb
```
3. Run the cells sequentially to preprocess data, train the model, and evaluate its performance.

## Model Architecture
The CNN model comprises the following layers:
- Convolutional layers with ReLU activation
- MaxPooling layers to reduce dimensionality
- Fully connected dense layers for classification
- Softmax activation for the final layer

## Results
The model achieves an accuracy of **82%** on the test dataset. (Replace **85** with the actual performance metric after running the notebook.)

