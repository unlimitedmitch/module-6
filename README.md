# Fashion MNIST Classification

## Project Overview
This project implements a Convolutional Neural Network (CNN) to classify images from the Fashion MNIST dataset. It's designed as part of a machine learning research task to classify fashion items, which could be used in targeting marketing for different products.

## Features
- Loads and preprocesses the Fashion MNIST dataset
- Implements a 6-layer CNN using Keras
- Trains the model on the dataset
- Evaluates the model's performance
- Makes predictions on sample images

## Requirements
- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib

## Installation
1. Ensure you have Python 3.7 or later installed.
2. Install the required libraries:


## Usage
1. Clone this repository or download the Python script.
2. Run the script:

The script will automatically download the Fashion MNIST dataset, train the model, and display results.

## Code Explanation

### Importing Libraries
```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
```
This section imports the necessary libraries: TensorFlow for the neural network, NumPy for numerical operations, and Matplotlib for plotting.

```python
(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
```

This code loads the Fashion MNIST dataset, normalizes the pixel values to be between 0 and 1, and reshapes the images to include a channel dimension.

```python
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
```
This creates a sequential model with three convolutional layers, two max pooling layers, and two dense layers. The final layer has 10 neurons, one for each class in the dataset.

```python
history = model.fit(train_images, train_labels, epochs=10, 
                    validation_split=0.2)
```
This code trains the model on the training data for 10 epochs, using 20% of the training data for validation.

```python
predictions = model.predict(np.array([image1, image2]))
```
This uses the trained model to make predictions on two sample images.


## Model Architecture
The CNN model consists of 6 layers:
1. Convolutional layer (32 filters)
2. Max pooling layer
3. Convolutional layer (64 filters)
4. Max pooling layer
5. Convolutional layer (64 filters)
6. Dense layer (64 neurons)
Output layer: Dense layer (10 neurons, one for each class)

## Dataset
The Fashion MNIST dataset consists of 70,000 grayscale images in 10 categories. The images show individual articles of clothing at low resolution (28 by 28 pixels).

## Results
The script will output:
- Model summary
- Training progress
- Final test accuracy
- Predictions on two sample images (displayed graphically)

## Future Improvements
- Experiment with different model architectures
- Implement data augmentation to improve model generalization
- Try transfer learning with pre-trained models