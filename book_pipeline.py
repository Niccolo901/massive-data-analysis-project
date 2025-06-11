import tensorflow as tf
import time
import psutil
import logging
import os
import pandas as pd
import aiohttp
import asyncio
from PIL import Image
from io import BytesIO
import ast
from sklearn.model_selection import train_test_split

# Disable oneDNN optimizations (optional)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ========== Data Augmentation ==========
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ========== IMAGE DOWNLOAD + LABEL ENRICHMENT ==========
def clean_title(title):
    """Sanitize book title for filename use."""
    return ''.join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in title).strip().replace(' ', '_')


# Map: Download and preprocess image
async def map_download_image(session, idx, title, url, output_dir, retries=3):
    """Download and validate an image asynchronously."""
    for attempt in range(retries):
        try:
            async with session.get(url) as response:
                response.raise_for_status()
                content = await response.read()

                # Validate image
                image = Image.open(BytesIO(content))
                image = image.convert("RGB")  # Ensure RGB format
                image = image.resize((224, 224))  # Resize for CNN input

            filename = os.path.join(output_dir, f"{idx}_{clean_title(title)}.jpg")
            image.save(filename)  # Save the preprocessed image
            return idx, filename

        except Exception as e:
            if attempt == retries - 1:
                return idx, None


# MapReduce-style processing
async def process_chunk(chunk, output_dir):
    """Process a chunk of the dataset."""
    async with aiohttp.ClientSession() as session:
        tasks = [
            map_download_image(session, idx, row['Title'], row['image'], output_dir)
            for idx, row in chunk.iterrows()
        ]
        return await asyncio.gather(*tasks)


def reduce_aggregate_results(results, csv_path, output_file):
    """Join image paths back into the dataframe and save."""
    df = pd.read_csv(csv_path)
    df['image_path'] = None
    for idx, path in results:
        if path:
            df.at[idx, 'image_path'] = path
    df = df.dropna(subset=['image_path'])
    df.to_csv(output_file, index=False)


def enrich_with_labels(input_csv="books_with_covers.csv", output_csv="books_with_covers_optimized.csv"):
    """Adds label column based on the first category in 'categories'."""
    df = pd.read_csv(input_csv)

    # Normalize image paths
    df['image_path'] = df['image_path'].str.replace("\\", "/", regex=False)

    # Extract and encode main category
    df['categories'] = df['categories'].fillna("[]")
    df['main_category'] = df['categories'].apply(lambda x: ast.literal_eval(x)[0] if ast.literal_eval(x) else "Unknown")
    df['label'] = df['main_category'].astype('category').cat.codes

    df = df.dropna(subset=['image_path'])
    df.to_csv(output_csv, index=False)
    print(f" {output_csv} created with 'label' column.")


def mapreduce_process(csv_path, output_dir, output_file, chunk_size=1000, max_images=None):
    os.makedirs(output_dir, exist_ok=True)
    results = []

    row_count = 0
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        if max_images is not None and row_count >= max_images:
            break

        if max_images is not None:
            remaining = max_images - row_count
            chunk = chunk.iloc[:remaining]

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        chunk_results = loop.run_until_complete(process_chunk(chunk, output_dir))
        results.extend(chunk_results)

        row_count += len(chunk)

    reduce_aggregate_results(results, csv_path, output_file)


# Entry point
def run_download():
    csv_path = 'dataset/books_subdata.csv'  #Subset of books data
    output_dir = 'book_covers'
    intermediate_file = 'books_with_covers.csv'
    final_output_file = 'books_with_covers_optimized.csv'

    mapreduce_process(csv_path, output_dir, intermediate_file)
    enrich_with_labels(intermediate_file, final_output_file)



# ========== TENSORFLOW DATA PIPELINE ==========
def preprocess_image(image_path):
    """Load and preprocess an image."""
    logging.info(f"Preprocessing image: {image_path}")
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)  # Decode JPEG
    image = tf.image.resize(image, [224, 224])  # Resize to 224x224
    image = tf.image.convert_image_dtype(image, tf.float32)  # Normalize to [0, 1]
    return image

@tf.autograph.experimental.do_not_convert
def create_dataset(csv_path, batch_size, augment=False, max_images=None):
    df = pd.read_csv(csv_path)
    df['image_path'] = df['image_path'].str.replace("\\", "/", regex=False)
    if max_images:
        df = df.iloc[:max_images]

    image_paths = df['image_path'].tolist()
    labels = df['label'].tolist()

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    def load_and_preprocess(image_path, label):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [224, 224])
        image = tf.image.convert_image_dtype(image, tf.float32)
        if augment:
            image = data_augmentation(image)
        return image, label

    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset



def run_preprocessing():
    csv_path = 'books_with_covers_optimized.csv'
    batch_size = 64  # Adjust batch size based on available memory
    max_images = 1000  # Limit the number of images to process

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


def split_dataset(csv_path, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['image_path', 'label'])

    # Filter out labels with fewer than 3 total samples
    label_counts = df['label'].value_counts()
    eligible_labels = label_counts[label_counts >= 3].index
    df = df[df['label'].isin(eligible_labels)]

    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df, train_size=train_size, stratify=df['label'], random_state=random_state)

    # Second split: val vs test
    rel_val_size = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df, train_size=rel_val_size, stratify=temp_df['label'], random_state=random_state)

    return train_df, val_df, test_df


def create_dataset_from_df(df, batch_size, augment=False):
    image_paths = df['image_path'].tolist()
    labels = df['label'].tolist()
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    def load_and_preprocess(image_path, label):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [224, 224])
        image = tf.image.convert_image_dtype(image, tf.float32)
        if augment:
            image = data_augmentation(image)
        return image, label

    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


if __name__ == "__main__":
    # For testing or CLI use only:
    run_download(max_images=300)

