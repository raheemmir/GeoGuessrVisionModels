import pandas as pd
import matplotlib.pyplot as plt
import os

out_dir = "evaluation/plots"

def pie_chart_distribution(csv, n, sizes_col, split):
    df = pd.read_csv(csv)
    top_df = df.iloc[:n]
    others_count = df.iloc[n:][sizes_col].sum()

    x = top_df[sizes_col].to_list() + [others_count]
    labels = top_df["country"].to_list() + ["Other"]

    plt.figure(figsize=(8, 8))
    plt.pie(
        x=x,
        labels=labels,
        autopct="%1.1f%%",
        pctdistance=0.9
    )

    plt.title(f"{split}-set country distributions (top-{n} labeled)")
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"{split}_country_distributions_top_{n}.png"), dpi=300)
    plt.show()

def main():
    csv_paths = {"train": "dataset/dataset_construction/train_quotas.csv", "test": "dataset/dataset_construction/test_quotas.csv"}
    sizes_col = "quota_ceil"
    n = 15
    for split, path in csv_paths.items():
        pie_chart_distribution(path, n, sizes_col, split)

if __name__ == "__main__":
    main()
