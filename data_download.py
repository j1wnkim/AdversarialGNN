import os
import requests

# Base URL for the Facebook dataset
BASE_URL = "https://raw.githubusercontent.com/benedekrozemberczki/karateclub/master/dataset/node_level/facebook"
FILES = ["edges.csv", "features.csv", "target.csv"]

# Directory paths
ROOT_DIR = "./datasets"
DATASET_NAME = "facebook"
RAW_DIR = os.path.join(ROOT_DIR, DATASET_NAME, "raw")

# Ensure the directory exists
os.makedirs(RAW_DIR, exist_ok=True)

# Function to download a file
def download_file(url, dest_path):
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(dest_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        print(f"Saved to {dest_path}")
    else:
        print(f"Failed to download {url}. HTTP Status Code: {response.status_code}")

# Download each required file
for file_name in FILES:
    file_url = f"{BASE_URL}/{file_name}"
    dest_path = os.path.join(RAW_DIR, file_name)
    download_file(file_url, dest_path)

print("All files downloaded and organized.")
