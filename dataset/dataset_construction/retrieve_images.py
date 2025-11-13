import os
import pandas as pd
from huggingface_hub import hf_hub_download
from zipfile import ZipFile
from pathlib import Path
import shutil

def retrieve_images(path_to_dataset_csv: str, split: str, output_parent_dir: str, repo_id: str = "osv5m/osv5m", repo_type: str ="dataset"):
    dataset_df = pd.read_csv(path_to_dataset_csv)
    images_to_extract = set(dataset_df["id"].astype(str) + ".jpg")

    total_images_retrieved = 0
    zip_files_processed = 0
    output_dir = os.path.join(output_parent_dir, split)
    print(f"Saving images to (output directory): {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    while images_to_extract:
        for i in range(98):
            filename = f"images/{split}/{i:02d}.zip"
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

                os.remove(zip_path) 
                if not os.path.exists(zip_path):
                    print(f"Zip {filename} successfully deleted")

                if (zip_files_processed % 10 == 0):
                    cache_dir = Path.home() / ".cache" / "huggingface"

                    if cache_dir.exists() and cache_dir.is_dir():
                        print(f"Deleting cache directory: {cache_dir}")
                        shutil.rmtree(cache_dir)
                        print("Cleared the huggingface cache!")
                    else:
                        print("No cache directory to delete")

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
    print(f"Total images retrieved: {total_images_retrieved}")
    print(f"Number of zip files processed to retrieve sampled dataset: {zip_files_processed}")

    print("Verification:")
    expected_images = set(dataset_df["id"].astype(str) + ".jpg")
    print(f"Total expected images (based on csv manifest): {len(expected_images)}")
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

def main(): 
    train_dataset_csv = "dataset/dataset_construction/sampled_train_dataset.csv"
    test_dataset_csv = "dataset/dataset_construction/sampled_test_dataset.csv"
    output_parent_dir = "dataset/images"
    # retrieve_images(train_dataset_csv, "train", output_parent_dir)
    retrieve_images(test_dataset_csv, "test", output_parent_dir)

if __name__ == "__main__":
    main()