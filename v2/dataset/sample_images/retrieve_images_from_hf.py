import os
import pandas as pd
from huggingface_hub import hf_hub_download
from zipfile import ZipFile

SAMPLED_TRAIN_METADATA_IN = "v2/assets/sampled_train_metadata.csv"
SAMPLED_VAL_METADATA_IN = "v2/assets/sampled_val_metadata.csv"
SAMPLED_TEST_METADATA_IN = "v2/assets/sampled_test_metadata.csv"
SAMPLED_TRAIN_METADATA_OUT = "v2/assets/sampled_train_metadata_cleaned.csv"
SAMPLED_VAL_METADATA_OUT = "v2/assets/sampled_val_metadata_cleaned.csv"
SAMPLED_TEST_METADATA_OUT = "v2/assets/sampled_test_metadata_cleaned.csv"
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
TEST_SPLIT = "test"
IMAGES_PARENT_DIR = "v2/assets/images"
COLUMNS_TO_KEEP = ["id", "latitude", "longitude", "country", "region", "sub-region", "city"]


def retrieve_images(metadata_csv: str, split: str, metadata_save_path: str, repo_id: str = "osv5m/osv5m", repo_type: str ="dataset"):
    df = pd.read_csv(metadata_csv, usecols=COLUMNS_TO_KEEP)
    images_to_extract = set(df["id"].astype(str) + ".jpg")

    total_images_retrieved = 0
    zip_files_processed = 0
    output_dir = os.path.join(IMAGES_PARENT_DIR, split)
    print(f"Saving images to (output directory): {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    while images_to_extract:
        for i in range(98):
            filename = f"images/{split}/{i:02d}.zip" if split != VAL_SPLIT else f"images/{TRAIN_SPLIT}/{i:02d}.zip"
            print(f"Processing {filename}")
            print(f"Images left to retrieve: {len(images_to_extract)}")

            try:
                zip_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)
                print(f"Path to current zip {filename}: {zip_path}")
                with ZipFile(zip_path, "r") as zip_file:
                    zip_files = zip_file.namelist() 
                    zip_images = [f for f in zip_files if f.endswith('.jpg')]
                    map_basename_to_fullpath = {os.path.basename(f): f for f in zip_images}
                    zip_images_basenames = set(map_basename_to_fullpath.keys())
                    extract_from_this_zip = images_to_extract.intersection(zip_images_basenames)

                    if extract_from_this_zip:
                        print(f"Number of images found in this zip ({filename}): {len(extract_from_this_zip)}")
                        for img in extract_from_this_zip:
                            full_path_img = map_basename_to_fullpath[img]
                            zip_file.extract(full_path_img, output_dir)
                            print(f"retrieved {full_path_img}")
                            total_images_retrieved += 1
                        print(f"Retrieved {len(extract_from_this_zip)} images from {filename}")
                        images_to_extract.difference_update(extract_from_this_zip)
                    else:
                        print(f"No images in this zip {filename}")
                
                zip_files_processed += 1

                if not images_to_extract:
                    print("Successfully retrieved all images, stopping now!")
                    break

            except Exception as e:
                print(f"Exception thrown: {e}")

    # Flatten the output_dir's subfolders (occurs due to .extract behaviour / design of osv5m HF zips)
    for subdir in os.listdir(output_dir):
        subdir_path = os.path.join(output_dir, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.endswith(".jpg"):
                    src = os.path.join(subdir_path, file)
                    dst = os.path.join(output_dir, file)
                    os.rename(src, dst)
            os.rmdir(subdir_path)
            print(f"Successfully flattened/removed {subdir}")
    
    # Verification
    print("Verification:")
    print(f"Total images retrieved: {total_images_retrieved}")
    print(f"Number of zip files processed to retrieve sampled dataset: {zip_files_processed}")
    expected_images = set(df["id"].astype(str) + ".jpg")
    print(f"Total expected images: {len(expected_images)}")
    actual_images_filter = [f for f in os.listdir(output_dir) if f.endswith(".jpg")]
    actual_images = set(actual_images_filter)
    print(f"Total actual images in the output directory: {len(actual_images)}")

    if expected_images == actual_images:
        print("Verification passed! All expected images have been retrieved!")
    else:
        print("Verification failed! There are mismatches!")
        missing = expected_images - actual_images
        extra = actual_images - expected_images
        print(f"There are {len(missing)} missing images from the dataset")
        print(missing)
        print(f"There are {len(extra)} extra images in the dataset")
        print(extra)

    # Add image paths / save the updated metadata csv
    df["image_paths"] = df["id"].astype(str).apply(
        lambda x: os.path.join(output_dir, x + ".jpg")
    )
    df.to_csv(metadata_save_path, index=False)
    print("Saved an updated metadata csv to: " + metadata_save_path)

def main(): 
    # retrieve_images(SAMPLED_TRAIN_METADATA_IN, TRAIN_SPLIT, SAMPLED_TRAIN_METADATA_OUT)
    # retrieve_images(SAMPLED_TEST_METADATA_IN, TEST_SPLIT, SAMPLED_TEST_METADATA_OUT)
    retrieve_images(SAMPLED_VAL_METADATA_IN, VAL_SPLIT, SAMPLED_VAL_METADATA_OUT)

if __name__ == "__main__":
    main()