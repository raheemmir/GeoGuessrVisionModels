import pandas as pd
import os

input_csv = "dataset/dataset_construction/sampled_test_dataset.csv"
output_root = "dataset/metadata/processed/v1_with_paths/"
output_csv = "dataset/metadata/processed/v1_with_paths/test_metadata.csv"
images_root = "dataset/images/test"
img_file_ext = ".jpg"
columns_to_keep = ["id", "latitude", "longitude", "country", "region", "sub-region", "city"]

def get_image_path(img_id):
    filename = str(img_id) + img_file_ext
    file_path = os.path.join(images_root, filename)
    return file_path

def check_if_exists(img_path):
    return os.path.isfile(img_path)

def add_img_paths():
    df = pd.read_csv(input_csv, usecols=columns_to_keep)
    print("Dataframe shape and columns:")
    print(df.shape)
    print(df.columns)

    df["image_paths"] = df["id"].apply(get_image_path)
    df["image_exists"] = df["image_paths"].apply(check_if_exists)
    missing_mask = df["image_exists"] == False
    print(f"Total rows: {len(df)}, Total missing image files: {missing_mask.sum()}")

    if missing_mask.sum() == 0:
        print("Successfully located all image paths, saving v1 version of dataset csv...")
        df.drop(columns=["image_exists"], inplace=True) # no longer needed
        print("Final Dataframe columns:")
        print(df.columns)
        os.makedirs(output_root, exist_ok=True)
        df.to_csv(output_csv, index=False)
    else:
        print(f"{missing_mask.sum()} image paths were not found! Something went wrong...")

def main():
    add_img_paths()

if __name__ == "__main__":
    main()
