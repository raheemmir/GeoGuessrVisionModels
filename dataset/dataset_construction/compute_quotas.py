import pandas as pd
import math

def compute_quotas(country_distribution_csv: str, target_total: int, output_filename: str):
    df = pd.read_csv(country_distribution_csv)
    df["raw_quota"] = df["proportion"] * target_total
    df["quota_ceil"] = df["raw_quota"].apply(math.ceil) # overshooting slightly
    target_minus_quota_ceil = target_total - df["quota_ceil"].sum()
    print(f"Target total: {target_total}, total of quota_ceil column: {df['quota_ceil'].sum()}")
    print(f"|Target quota - ceiling quota| = |{abs(target_minus_quota_ceil)}|")
    print("Dataframe:")
    print(df.head(25))

    df.to_csv(f"dataset/dataset_construction/{output_filename}.csv", index=False)


def main():
    # Aiming for a dataset of 150k images with an 80/20 train-test split
    train_distributions_csv = "dataset/country_distributions/train_split_country_distributions.csv"
    train_target_total = 120000
    compute_quotas(train_distributions_csv, train_target_total, "train_quotas")
    test_distributions_csv = "dataset/country_distributions/test_split_country_distributions.csv"
    test_target_total = 30000
    compute_quotas(test_distributions_csv, test_target_total, "test_quotas")
    

if __name__ == "__main__":
    main()

