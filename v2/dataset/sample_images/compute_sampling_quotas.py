import pandas as pd
import math

TRAIN_DISTRIBUTIONS = "v2/assets/train_split_country_distributions_filtered.csv"
TEST_DISTRIBUTIONS = "v2/assets/test_split_country_distributions_filtered.csv"
TRAIN_QUOTA = 180000
VAL_QUOTA = 22500
TEST_QUOTA = 22500
TRAIN_OUTPUT = "v2/assets/train_quotas.csv"
VAL_OUTPUT = "v2/assets/val_quotas.csv"
TEST_OUTPUT = "v2/assets/test_quotas.csv"

def compute_sampling_quotas(distribution_csv: str, target_total: int, output_path: str):
    df = pd.read_csv(distribution_csv)
    df["raw_quota"] = df["proportion"] * target_total
    df["quota"] = df["raw_quota"].apply(math.ceil) # overshooting slightly
    target_minus_actual_quota = target_total - df["quota"].sum()
    print(f"|Target Total - Actual Total| = {abs(target_minus_actual_quota)}")
    print("Dataframe:")
    print(df.head(25))
    df.to_csv(output_path, index=False)

def main():
    compute_sampling_quotas(TRAIN_DISTRIBUTIONS, TRAIN_QUOTA, TRAIN_OUTPUT)
    compute_sampling_quotas(TRAIN_DISTRIBUTIONS, VAL_QUOTA, VAL_OUTPUT)
    compute_sampling_quotas(TEST_DISTRIBUTIONS, TEST_QUOTA, TEST_OUTPUT)

if __name__ == "__main__":
    main()