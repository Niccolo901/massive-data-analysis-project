import os
import kaggle
import zipfile

# Define Kaggle dataset identifier
DATASET_NAME = "mohamedbakhet/amazon-books-reviews"
DOWNLOAD_PATH = "dataset"  # Destination folder for extracted files

# Ensure the Kaggle API configuration directory exists
KAGGLE_CONFIG_DIR = os.path.expanduser("~/.kaggle")
os.environ["KAGGLE_CONFIG_DIR"] = KAGGLE_CONFIG_DIR

# Verify Kaggle API credentials
kaggle_json_path = os.path.join(KAGGLE_CONFIG_DIR, "kaggle.json")
if not os.path.exists(kaggle_json_path):
    raise FileNotFoundError(f"Kaggle API key not found at {kaggle_json_path}. Please ensure it is correctly placed.")

# Download dataset
print("Downloading dataset from Kaggle...")
kaggle.api.dataset_download_files(DATASET_NAME, path=DOWNLOAD_PATH, unzip=True)
print(f"Dataset successfully downloaded and extracted to '{DOWNLOAD_PATH}'.")

# Verify downloaded files
downloaded_files = os.listdir(DOWNLOAD_PATH)
if not downloaded_files:
    raise RuntimeError("No files were extracted. Please check the dataset structure.")
else:
    print(f"Extracted files: {downloaded_files}")
