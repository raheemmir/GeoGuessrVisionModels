import pandas as pd

SAMPLED_TRAIN_METADATA = "v2/assets/sampled_train_metadata_cleaned.csv"
SAMPLED_VAL_METADATA = "v2/assets/sampled_val_metadata_cleaned.csv"
SAMPLED_TEST_METADATA = "v2/assets/sampled_test_metadata_cleaned.csv"

def main():
    train_df = pd.read_csv(SAMPLED_TRAIN_METADATA)
    val_df = pd.read_csv(SAMPLED_VAL_METADATA)
    test_df = pd.read_csv(SAMPLED_TEST_METADATA)

    train_df["image_paths"] = train_df["image_paths"].str.replace("\\", "/", regex=False)
    val_df["image_paths"] = val_df["image_paths"].str.replace("\\", "/", regex=False)
    test_df["image_paths"] = test_df["image_paths"].str.replace("\\", "/", regex=False)

    train_df.to_csv(SAMPLED_TRAIN_METADATA, index=False)
    val_df.to_csv(SAMPLED_VAL_METADATA, index=False)
    test_df.to_csv(SAMPLED_TEST_METADATA, index=False)

if __name__ == "__main__":
    main()