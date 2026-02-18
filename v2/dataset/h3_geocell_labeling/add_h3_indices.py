import pandas as pd
import h3
import os

SAMPLED_TRAIN_METADATA = "v2/assets/sampled_train_metadata_cleaned.csv"
SAMPLED_VAL_METADATA = "v2/assets/sampled_val_metadata_cleaned.csv"
SAMPLED_TEST_METADATA = "v2/assets/sampled_test_metadata_cleaned.csv"
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
TEST_SPLIT = "test"
OUTPUT_ROOT = "v2/assets/"

def add_h3_index_column(metadata_csv, split_name, r):
    df = pd.read_csv(metadata_csv)
    print("Dataframe shape and columns:")
    print(df.shape)
    print(df.columns)

    h3_indices = []
    for i in range(len(df)):
        lat = df.iloc[i]["latitude"]
        lon = df.iloc[i]["longitude"]
        h3_index = h3.latlng_to_cell(lat, lon, r)
        h3_indices.append(h3_index)

    df[f"h3_r{r}"] = h3_indices

    # verification
    incorrect_h3_count = 0
    for i in range(len(df)):
        lat = df.iloc[i]["latitude"]
        lon = df.iloc[i]["longitude"]
        verifying_h3 = h3.latlng_to_cell(lat, lon, r)
        if verifying_h3 != df.iloc[i][f"h3_r{r}"]:
            incorrect_h3_count += 1
    
    assert len(h3_indices) == len(df), "Length mismatch between H3 column and Dataframe!"
    assert incorrect_h3_count == 0, f"{incorrect_h3_count} mismatches found!"

    print("Updated Dataframe shape and columns:")
    print(df.shape)
    print(df.columns)
    
    output_path = os.path.join(OUTPUT_ROOT, f"{split_name}_metadata_h3_r{r}.csv")
    df.to_csv(output_path, index=False)
    print(f"saved -> {output_path}")

def main():
    dataset_splits = {
        TRAIN_SPLIT: SAMPLED_TRAIN_METADATA, 
        VAL_SPLIT: SAMPLED_VAL_METADATA, 
        TEST_SPLIT: SAMPLED_TEST_METADATA
    }
    resolutions = [1, 2]

    for split_name, metadata_csv in dataset_splits.items():
        for r in resolutions:
            add_h3_index_column(metadata_csv=metadata_csv, split_name=split_name, r=r)

if __name__== "__main__":
    main()