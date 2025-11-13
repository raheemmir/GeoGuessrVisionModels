import pandas as pd
import h3
from statistics import median
import numpy as np

train_csv = "dataset/metadata/processed/v1_with_paths/train_metadata.csv"
test_csv = "dataset/metadata/processed/v1_with_paths/test_metadata.csv"
resolutions = [1, 2, 3] # [1, 2, 3, 4, 5]

def analyze_resolutions():
    print("Analyzing h3 resolutions...")
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    total_imgs = len(train_df) # should be 120121
    print(f"Total train images: {total_imgs}")
    for r in resolutions:
        print(f"Resolution: {r}")
        counts = {}
        for i in range(len(train_df)):
            lat = train_df.iloc[i]["latitude"]
            lon = train_df.iloc[i]["longitude"]
            h3_index = h3.latlng_to_cell(lat, lon, r)
            if h3_index not in counts:
                counts[h3_index] = 1
            else:
                counts[h3_index] += 1
        
        num_labels = len(counts) # how many h3 cells
        counts_per_label = list(counts.values())
        
        avg_imgs_per_label = total_imgs / num_labels
        min_imgs_per_label = min(counts.values())
        max_imgs_per_label = max(counts.values())

        median_imgs_per_label = median(counts_per_label)
        p10 = np.percentile(np.array(counts_per_label), 10)
        p25 = np.percentile(np.array(counts_per_label), 25)
        p50 = np.percentile(np.array(counts_per_label), 50) # this is the median
        p75 = np.percentile(np.array(counts_per_label), 75)
        p90 = np.percentile(np.array(counts_per_label), 90)
        
        # Percentage of test samples belonging to h3 cells not in training set
        test_samples_with_ids_not_in_training = calculate_unseen_data_percentage(test_df, set(counts.keys()), r)

        print(f"Number of geocell labels: {num_labels}")
        print(f"Average number of images per label: {avg_imgs_per_label}")
        print(f"Median number of images per label: {median_imgs_per_label}")
        print(f"Percentiles (images per class):")
        print(f"P10: {p10}")
        print(f"P25: {p25}")
        print(f"P50: {p50}")
        print(f"p75: {p75}")
        print(f"P90: {p90}")
        
        print(f"Minimum number of images in a label: {min_imgs_per_label}")
        print(f"Maximum number of images in a label: {max_imgs_per_label}")

        print(f"Percentage of test samples belonging to h3 cells not in training data: {test_samples_with_ids_not_in_training}")

def calculate_unseen_data_percentage(test_df, train_cells, res):
    # Answers the question:
    # "What percentage of test samples belong to h3 cells
    # that don't appear in training?"
    num_samples_with_unseen_labels = 0
    for i in range(len(test_df)):
        lat = test_df.iloc[i]["latitude"]
        lon = test_df.iloc[i]["longitude"]
        h3_index = h3.latlng_to_cell(lat, lon, res)
        if h3_index not in train_cells:
            num_samples_with_unseen_labels += 1

    prop_of_samples_with_unseen_labels = num_samples_with_unseen_labels / len(test_df)
    return prop_of_samples_with_unseen_labels * 100

def main():
    analyze_resolutions()

if __name__ == "__main__":
    main()







