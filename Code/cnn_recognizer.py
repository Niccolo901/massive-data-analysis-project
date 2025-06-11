import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

def create_cnn(input_shape=(224, 224, 3), num_classes=10, conv_filters=[32, 64, 128], dense_units=128, dropout_rate=0.3):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    for filters in conv_filters:
        model.add(layers.Conv2D(filters, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(dropout_rate))
    model.add(layers.Flatten())
    model.add(layers.Dense(dense_units, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

def create_transfer_model(input_shape=(224, 224, 3), num_classes=10, dropout_rate=0.3):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze base model for transfer learning
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def get_model(model_type="cnn", **kwargs):
    """Factory function to choose model architecture."""
    if model_type == "cnn":
        return create_cnn(**kwargs)
    elif model_type == "mobilenet":
        return create_transfer_model(**kwargs)
    else:
        raise ValueError("Unknown model_type. Choose 'cnn' or 'mobilenet'.")


# Optional: Enable distributed training (if using multiple GPUs)
# This requires TensorFlow to detect multiple devices, such as in a cloud or cluster environment.
#
# strategy = tf.distribute.MirroredStrategy()
# with strategy.scope():
#     model = get_model(model_type="mobilenet", input_shape=(224, 224, 3), num_classes=10)
#     model.compile(optimizer='adam',
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])


def train_model(model, train_data, val_data, epochs=20, batch_size=32):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    cb = [
        callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        callbacks.ModelCheckpoint('../best_model.keras', save_best_only=True, monitor='val_loss')
    ]
    history = model.fit(train_data,
                        validation_data=val_data,
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=cb)
    return model, history

def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train Accuracy')
    plt.plot(val_acc, label='Val Accuracy')
    plt.legend()
    plt.title('Accuracy over Epochs')
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.legend()
    plt.title('Loss over Epochs')
    plt.show()

def evaluate_model(model, test_data, test_labels=None):
    results = model.evaluate(test_data, return_dict=True)
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
    if test_labels is not None:
        preds = model.predict(test_data)
        y_pred = preds.argmax(axis=1)
        print(classification_report(test_labels, y_pred))
    return results