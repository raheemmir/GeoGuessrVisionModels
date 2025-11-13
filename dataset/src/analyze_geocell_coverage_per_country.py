import pandas as pd
import h3
import os

train_csv = "dataset/metadata/processed/v1_with_paths/train_metadata.csv"
test_csv = "dataset/metadata/processed/v1_with_paths/test_metadata.csv"
output_root = "dataset/metadata/processed/analysis"
resolution = 2

def analyze_geocell_country_coverage(input_csv, split_name):
    df = pd.read_csv(input_csv)
    country_stats = {}
    for i in range(len(df)):
        country = df.iloc[i]["country"]
        lat = df.iloc[i]["latitude"]
        lon = df.iloc[i]["longitude"]
        h3_index = h3.latlng_to_cell(lat, lon, resolution)
        if country not in country_stats:
            country_stats[country] = {"cells": set(), "num_images": 0}
        country_stats[country]["cells"].add(h3_index)
        country_stats[country]["num_images"] += 1

    output_rows = []
    for country, stats in country_stats.items():
        num_cells = len(stats["cells"])
        num_images = stats["num_images"]
        avg_imgs_per_cell = num_images / num_cells
        output_rows.append({
            "country": country,
            "num_cells": num_cells,
            "total_images": num_images,
            "avg_images_per_cell": avg_imgs_per_cell
        })

    output_df = pd.DataFrame(output_rows).sort_values("total_images", ascending=False)
    print(f"Top 20 countries by image count: ")
    print(output_df.head(20).to_string(index=False))

    os.makedirs(output_root, exist_ok=True)
    output_filename = f"h3_r{resolution}_country_stats_{split_name}.csv"
    output_path = os.path.join(output_root, output_filename)
    output_df.to_csv(output_path, index=False)
    print(f"[{split_name}] saved -> {output_path}")

def main():
    analyze_geocell_country_coverage(train_csv, "train")
    analyze_geocell_country_coverage(test_csv, "test")

if __name__ == "__main__":
    main()



        


