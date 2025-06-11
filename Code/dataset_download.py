import os
import kaggle
import zipfile


def download_dataset(dataset_name="mohamedbakhet/amazon-books-reviews",
                     download_path="dataset",
                     expected_file="Books_data.csv"):
    """
    Downloads and extracts a dataset from Kaggle if not already present.

    Parameters:
        dataset_name (str): Kaggle dataset identifier.
        download_path (str): Local directory to store the dataset.
        expected_file (str): A key file expected to exist after extraction.

    Returns:
        str: Path to the expected file.
    """
    kaggle_config_dir = os.path.expanduser("~/.kaggle")
    kaggle_json_path = os.path.join(kaggle_config_dir, "kaggle.json")

    if not os.path.exists(kaggle_json_path):
        raise FileNotFoundError(
            f"Kaggle API key not found at {kaggle_json_path}. "
            "Make sure to upload your kaggle.json to authenticate with the Kaggle API."
        )

    os.environ["KAGGLE_CONFIG_DIR"] = kaggle_config_dir

    expected_path = os.path.join(download_path, expected_file)
    if os.path.exists(expected_path):
        print(f"Dataset already exists at '{expected_path}'. Skipping download.")
        return expected_path

    print("Downloading dataset from Kaggle...")
    kaggle.api.dataset_download_files(dataset_name, path=download_path, unzip=True)
    print(f"Dataset successfully downloaded and extracted to '{download_path}'.")

    # Check for unzipped or leftover .zip files
    for file in os.listdir(download_path):
        if file.endswith(".zip"):
            zip_path = os.path.join(download_path, file)
            print(f"Extracting zip file: {zip_path}")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(download_path)
            os.remove(zip_path)

    # Final verification
    if not os.path.exists(expected_path):
        raise RuntimeError(f"Expected file '{expected_file}' not found in '{download_path}'.")

    print(f"Dataset ready at '{expected_path}'.")
    return expected_path
