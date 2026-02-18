import pandas as pd
import json
import os

TRAIN_METADATA_H3_R1 = "v2/assets/train_metadata_h3_r1.csv"
VAL_METADATA_H3_R1 = "v2/assets/val_metadata_h3_r1.csv"
TRAIN_METADATA_H3_R2 = "v2/assets/train_metadata_h3_r2.csv"
VAL_METADATA_H3_R2 = "v2/assets/val_metadata_h3_r2.csv"
OUTPUT_ROOT = "v2/assets/"

def add_id_column_train(csv, r):
    df = pd.read_csv(csv)
    print(f"{csv} shape: {df.shape}")
    print(f"{csv} columns: {df.columns}")

    unique_h3_cells = sorted(df[f"h3_r{r}"].unique())
    h3_to_id = {}
    for i, cell in enumerate(unique_h3_cells):
        h3_to_id[cell] = i
    id_to_h3 = {}
    for i, cell in enumerate(unique_h3_cells):
        id_to_h3[i] = cell
    df[f"id_h3_r{r}"] = df[f"h3_r{r}"].map(h3_to_id)
    df[f"id_h3_r{r}"] = df[f"id_h3_r{r}"].astype(int)

    print(f"Number of unique labels: {len(unique_h3_cells)}")
    print(f"Updated shape: {df.shape}")
    print(f"Updated columns: {df.columns}")

    output_path = os.path.join(OUTPUT_ROOT, f"train_metadata_h3_r{r}_with_ids.csv")
    df.to_csv(output_path, index=False)

    h3_to_id_json = os.path.join(OUTPUT_ROOT, f"h3_r{r}_to_id.json")
    with open(h3_to_id_json, "w") as f:
        json.dump(h3_to_id, f)
    
    id_to_h3_json = os.path.join(OUTPUT_ROOT, f"id_to_h3_r{r}.json")
    with open(id_to_h3_json, "w") as f:
        json.dump(id_to_h3, f)

def add_id_column_val(csv, r):
    h3_to_id_path = os.path.join(OUTPUT_ROOT, f"h3_r{r}_to_id.json")
    with open(h3_to_id_path, "r") as f:
        h3_to_id = json.load(f)

    df = pd.read_csv(csv)
    print(f"{csv} shape: {df.shape}")
    print(f"{csv} columns: {df.columns}")

    df[f"id_h3_r{r}"] = df[f"h3_r{r}"].map(h3_to_id)
    unseen_cells = df[f"id_h3_r{r}"].isna().sum()
    if unseen_cells:
        print(f"Dropping {unseen_cells} rows belonging to h3 indicies not seen in training set")
        df = df.dropna(subset=[f"id_h3_r{r}"]).copy()

    df[f"id_h3_r{r}"] = df[f"id_h3_r{r}"].astype(int)
    print(f"Updated shape: {df.shape}")
    print(f"Updated columns: {df.columns}")

    output_path = os.path.join(OUTPUT_ROOT, f"val_metadata_h3_r{r}_with_ids.csv")
    df.to_csv(output_path, index=False)

def main():

    h3_metadata_mappings = {
        1: [TRAIN_METADATA_H3_R1, VAL_METADATA_H3_R1],
        2: [TRAIN_METADATA_H3_R2, VAL_METADATA_H3_R2]
    }

    for resolution, (train_csv, val_csv) in h3_metadata_mappings.items():
        add_id_column_train(csv=train_csv, r=resolution)
        add_id_column_val(csv=val_csv, r=resolution)

if __name__ == "__main__":
    main()