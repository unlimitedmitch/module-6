import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape images to include channel dimension
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)


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

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_split=0.2)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# Choose two images from the test set
image1 = test_images[0]
image2 = test_images[1]

# Make predictions
predictions = model.predict(np.array([image1, image2]))

# Get the predicted class
predicted_class1 = np.argmax(predictions[0])
predicted_class2 = np.argmax(predictions[1])

# Define class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(f"Prediction for image 1: {class_names[predicted_class1]}")
print(f"Prediction for image 2: {class_names[predicted_class2]}")

# Display the images
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.imshow(image1.reshape(28,28), cmap='gray')
plt.title(f"Predicted: {class_names[predicted_class1]}")
plt.subplot(122)
plt.imshow(image2.reshape(28,28), cmap='gray')
plt.title(f"Predicted: {class_names[predicted_class2]}")
plt.show()
