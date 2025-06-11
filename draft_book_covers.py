import os
import pandas as pd
import aiohttp
import asyncio
from PIL import Image
from io import BytesIO


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
    df['main_category'] = df['categories'].apply(lambda x: eval(x)[0] if eval(x) else "Unknown")
    df['label'] = df['main_category'].astype('category').cat.codes

    df = df.dropna(subset=['image_path'])
    df.to_csv(output_csv, index=False)
    print(f"✅ {output_csv} created with 'label' column.")


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

    reduce_aggregate_results(results, csv_path, output_file)


# Entry point
def run_download(max_images=100):
    csv_path = 'dataset/Books_data.csv'
    output_dir = 'book_covers'
    intermediate_file = 'books_with_covers.csv'
    final_output_file = 'books_with_covers_optimized.csv'

    mapreduce_process(csv_path, output_dir, intermediate_file, max_images=max_images)
    enrich_with_labels(intermediate_file, final_output_file)


if __name__ == "__main__":
    run_download()
