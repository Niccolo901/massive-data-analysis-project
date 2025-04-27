import tensorflow as tf
from tensorflow.keras import layers, models

def create_cnn(input_shape=(224, 224, 3), num_classes=10):
    """Create a CNN model for image recognition."""
    model = models.Sequential()

    # Convolutional Layer 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    # Convolutional Layer 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Convolutional Layer 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten and Fully Connected Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

# Example usage
if __name__ == "__main__":
    input_shape = (224, 224, 3)  # Image dimensions
    num_classes = 10  # Number of categories

    cnn_model = create_cnn(input_shape, num_classes)
    cnn_model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    # Summary of the model
    cnn_model.summary()