"""
Fix data split - reorganize images to match train_data.csv and val_data.csv
"""
import os
import shutil
import pandas as pd

# Paths
base_dir = "C:/Users/thc1006/Desktop/114-1/nycu-CSIC30014-LAB3"
train_csv = os.path.join(base_dir, "data/train_data.csv")
val_csv = os.path.join(base_dir, "data/val_data.csv")

train_images_dir = os.path.join(base_dir, "train_images")
val_images_dir = os.path.join(base_dir, "val_images")
test_images_dir = os.path.join(base_dir, "test_images")

# Collect all current images from all directories
all_images = {}
for dir_path in [train_images_dir, val_images_dir]:
    if os.path.exists(dir_path):
        for fname in os.listdir(dir_path):
            if fname.endswith(('.jpg', '.jpeg', '.png')):
                all_images[fname] = os.path.join(dir_path, fname)

print(f"Found {len(all_images)} total images across all directories")

# Load CSVs
train_df = pd.read_csv(train_csv)
val_df = pd.read_csv(val_csv)

print(f"Train CSV: {len(train_df)} entries")
print(f"Val CSV: {len(val_df)} entries")

# Function to ensure image is in correct directory
def ensure_image_location(fname, target_dir):
    if fname not in all_images:
        print(f"WARNING: {fname} not found in any directory!")
        return False

    current_path = all_images[fname]
    target_path = os.path.join(target_dir, fname)

    if current_path != target_path:
        # Move the file
        print(f"Moving {fname} to {os.path.basename(target_dir)}/")
        shutil.move(current_path, target_path)
        all_images[fname] = target_path  # Update tracking

    return True

# Process train images
print("\nProcessing train images...")
missing_train = 0
for fname in train_df['new_filename']:
    if not ensure_image_location(fname, train_images_dir):
        missing_train += 1

# Process val images
print("\nProcessing val images...")
missing_val = 0
for fname in val_df['new_filename']:
    if not ensure_image_location(fname, val_images_dir):
        missing_val += 1

print(f"\nDone!")
print(f"Missing train images: {missing_train}")
print(f"Missing val images: {missing_val}")

# Verify counts
train_count = len([f for f in os.listdir(train_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
val_count = len([f for f in os.listdir(val_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])

print(f"\nFinal counts:")
print(f"train_images/: {train_count} files (CSV expects {len(train_df)})")
print(f"val_images/: {val_count} files (CSV expects {len(val_df)})")
