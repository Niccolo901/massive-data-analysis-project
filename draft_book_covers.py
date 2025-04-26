import os
import pandas as pd
import aiohttp
import asyncio
from aiofiles import open as aio_open
from PIL import Image
from io import BytesIO
import tensorflow as tf


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


async def process_chunk(chunk, output_dir):
    """Process a chunk of the dataset."""
    async with aiohttp.ClientSession() as session:
        tasks = [
            map_download_image(session, idx, row['Title'], row['image'], output_dir)
            for idx, row in chunk.iterrows()
        ]
        return await asyncio.gather(*tasks)


# Reduce: Aggregate results
def reduce_aggregate_results(results, csv_path, output_file):
    df = pd.read_csv(csv_path)
    df['image_path'] = None
    for idx, path in results:
        if path:
            df.at[idx, 'image_path'] = path
    df = df.dropna(subset=['image_path'])
    df.to_csv(output_file, index=False)


# Process dataset in chunks
async def process_chunk(chunk, output_dir):
    async with aiohttp.ClientSession() as session:
        tasks = [
            map_download_image(session, idx, row['Title'], row['image'], output_dir)
            for idx, row in chunk.iterrows()
        ]
        return await asyncio.gather(*tasks)


def mapreduce_process(csv_path, output_dir, output_file, chunk_size=1000):
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        chunk_results = loop.run_until_complete(process_chunk(chunk, output_dir))
        results.extend(chunk_results)

    reduce_aggregate_results(results, csv_path, output_file)


# Entry point
def main():
    csv_path = 'dataset/Books_data.csv'
    output_dir = 'book_covers'
    output_file = 'books_with_covers.csv'
    mapreduce_process(csv_path, output_dir, output_file)


if __name__ == "__main__":
    main()