import pandas as pd
import h3
import os

SAMPLED_TRAIN_METADATA = "v2/assets/sampled_train_metadata_cleaned.csv"
SAMPLED_VAL_METADATA = "v2/assets/sampled_val_metadata_cleaned.csv"
SAMPLED_TEST_METADATA = "v2/assets/sampled_test_metadata_cleaned.csv"
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
TEST_SPLIT = "test"
OUTPUT_ROOT = "v2/assets"

def analyze_geocell_country_coverage(input_csv, split_name, resolution):
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

    output_filename = f"h3_r{resolution}_country_stats_{split_name}.csv"
    output_path = os.path.join(OUTPUT_ROOT, output_filename)
    output_df.to_csv(output_path, index=False)
    print(f"[{split_name}] saved -> {output_path}")

def main():
    resolutions = [1, 2, 3]
    for r in resolutions:
        print(f"Split: {TRAIN_SPLIT}, Resolution: {r}")
        analyze_geocell_country_coverage(SAMPLED_TRAIN_METADATA, TRAIN_SPLIT, r)
        print(f"Split: {VAL_SPLIT}, Resolution: {r}")
        analyze_geocell_country_coverage(SAMPLED_VAL_METADATA, VAL_SPLIT, r)
        print(f"Split: {TEST_SPLIT}, Resolution: {r}")
        analyze_geocell_country_coverage(SAMPLED_TEST_METADATA, TEST_SPLIT, r)

if __name__ == "__main__":
    main()



        


