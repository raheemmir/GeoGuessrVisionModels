import pandas as pd
import h3
import os

resolution = 2
input_csv = "dataset/metadata/processed/v1_with_paths/test_metadata.csv"
output_root = f"dataset/metadata/processed/v2_h3_r{resolution}/"
output_csv = f"dataset/metadata/processed/v2_h3_r{resolution}/test_metadata.csv"

def add_h3_geocells_column():
    df = pd.read_csv(input_csv)
    print("Dataframe shape and columns:")
    print(df.shape)
    print(df.columns)

    h3_indices = []
    for i in range(len(df)):
        lat = df.iloc[i]["latitude"]
        lon = df.iloc[i]["longitude"]
        h3_index = h3.latlng_to_cell(lat, lon, resolution)
        h3_indices.append(h3_index)

    df[f"h3_r{resolution}"] = h3_indices

    # verification
    incorrect_h3_count = 0
    for i in range(len(df)):
        lat = df.iloc[i]["latitude"]
        lon = df.iloc[i]["longitude"]
        verifying_h3 = h3.latlng_to_cell(lat, lon, resolution)
        if verifying_h3 != df.iloc[i][f"h3_r{resolution}"]:
            incorrect_h3_count += 1
    
    assert len(h3_indices) == len(df), "Length mismatch between H3 column and Dataframe!"
    assert incorrect_h3_count == 0, f"{incorrect_h3_count} mismatches found!"

    print("Updated Dataframe shape and columns:")
    print(df.shape)
    print(df.columns)
    
    os.makedirs(output_root, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"saved -> {output_csv}")

def main():
    add_h3_geocells_column()

if __name__ == "__main__":
    main()


        
