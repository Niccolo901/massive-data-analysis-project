import tensorflow as tf
import pandas as pd
import os
import time
import psutil
import logging

# Disable oneDNN optimizations (optional)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def preprocess_image(image_path):
    """Load and preprocess an image."""
    logging.info(f"Preprocessing image: {image_path}")
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)  # Decode JPEG
    image = tf.image.resize(image, [224, 224])  # Resize to 224x224
    image = tf.image.convert_image_dtype(image, tf.float32)  # Normalize to [0, 1]
    return image

@tf.autograph.experimental.do_not_convert
def create_dataset(csv_path, batch_size, max_images=None):
    """Create a tf.data.Dataset for image preprocessing."""
    # Load the CSV
    logging.info(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    image_paths = df['image_path'].dropna().tolist()  # Filter valid image paths

    # Limit the number of images if max_images is specified
    if max_images:
        logging.info(f"Limiting dataset to {max_images} images.")
        image_paths = image_paths[:max_images]

    # Create a TensorFlow Dataset
    logging.info(f"Total images to process: {len(image_paths)}")
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def main():
    csv_path = 'books_with_covers_optimized.csv'
    batch_size = 64  # Adjust batch size based on available memory
    max_images = 500  # Limit the number of images to process

    # Create and process the dataset
    logging.info("Starting dataset creation...")
    dataset = create_dataset(csv_path, batch_size)
    process = psutil.Process(os.getpid())

    for batch_idx, batch in enumerate(dataset):
        start_time = time.time()

        # Simulate processing the batch
        logging.info(f"Processed batch {batch_idx + 1} with shape: {batch.shape}")

        # Log metrics
        elapsed_time = time.time() - start_time
        memory_usage = process.memory_info().rss / (1024 * 1024)  # Memory in MB
        logging.info(
            f"Batch {batch_idx + 1} processed in {elapsed_time:.2f} seconds. Memory usage: {memory_usage:.2f} MB.")

    logging.info("Dataset processing completed.")

if __name__ == "__main__":
    main()