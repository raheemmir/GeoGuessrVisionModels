import pandas as pd
import json

input_csv = "dataset/metadata/processed/v2_h3_r2/train_metadata.csv"
output_csv = "dataset/metadata/processed/v2_h3_r2/train_metadata_with_ids.csv"
h3_to_id_json = "dataset/metadata/processed/v2_h3_r2/h3_to_id.json"
id_to_h3_json = "dataset/metadata/processed/v2_h3_r2/id_to_h3.json"

def main():
    df = pd.read_csv(input_csv)
    print(f"{input_csv} shape: {df.shape}")
    print(f"{input_csv} columns: {df.columns}")
    unique_h3_cells = sorted(df["h3_r2"].unique())
    h3_to_id = {}
    for i, cell in enumerate(unique_h3_cells):
        h3_to_id[cell] = i
    id_to_h3 = {}
    for i, cell in enumerate(unique_h3_cells):
        id_to_h3[i] = cell
    df["class_id"] = df["h3_r2"].map(h3_to_id)

    print(f"Number of unique labels: {len(unique_h3_cells)}")
    print(f"Updated shape: {df.shape}")
    print(f"Updated columns: {df.columns}")

    df.to_csv(output_csv, index=False)

    with open(h3_to_id_json, "w") as f:
        json.dump(h3_to_id, f)
    
    with open(id_to_h3_json, "w") as f:
        json.dump(id_to_h3, f)


if __name__ == "__main__":
    main()