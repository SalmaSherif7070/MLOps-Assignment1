import os
import zipfile
import shutil
from pathlib import Path

DATA_DIR = Path("data/sketch2pokemon")
ARCHIVE_PATH = DATA_DIR / "archive.zip"
EXTRACT_DIR = DATA_DIR / "extracted"

# Unzip archive
print("Unzipping archive.zip...")
with zipfile.ZipFile(ARCHIVE_PATH, 'r') as zip_ref:
    zip_ref.extractall(EXTRACT_DIR)

# Remove old directories
for old_dir in ["train", "test"]:
    path = DATA_DIR / old_dir
    if path.exists():
        shutil.rmtree(path)

# Find dataset folder
dataset_folder = None
for item in EXTRACT_DIR.iterdir():
    if item.is_dir() and "pokemon" in item.name.lower():
        dataset_folder = item
        break

if not dataset_folder:
    print("Error: Pokemon dataset folder not found!")
    exit(1)

# Move paired directories
for folder_name in ["trainA", "trainB", "testA", "testB"]:
    src = dataset_folder / folder_name
    dst = DATA_DIR / folder_name
    
    if dst.exists():
        shutil.rmtree(dst)
    
    if src.exists():
        shutil.move(str(src), str(dst))
        count = len(os.listdir(dst))
        print(f"{folder_name}: {count} images")

# Clean up
shutil.rmtree(EXTRACT_DIR)

# Summary
print("\nData preparation complete:")
print(f"trainA: {len(os.listdir(DATA_DIR / 'trainA'))} images")
print(f"trainB: {len(os.listdir(DATA_DIR / 'trainB'))} images")
print(f"testA: {len(os.listdir(DATA_DIR / 'testA'))} images")
print(f"testB: {len(os.listdir(DATA_DIR / 'testB'))} images")
print(f"   💡 trainA paired with trainB, testA paired with testB")
