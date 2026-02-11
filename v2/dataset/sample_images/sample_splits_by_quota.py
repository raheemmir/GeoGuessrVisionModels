from huggingface_hub import hf_hub_download
import pandas as pd

SEED = 42
TRAIN_QUOTA = "v2/assets/train_quotas.csv"
VAL_QUOTA = "v2/assets/val_quotas.csv"
TEST_QUOTA = "v2/assets/test_quotas.csv"
HF_TRAIN = "train.csv"
HF_TEST = "test.csv"
SAMPLED_TRAIN = "v2/assets/sampled_train_metadata.csv"
SAMPLED_VAL = "v2/assets/sampled_val_metadata.csv"
SAMPLED_TEST = "v2/assets/sampled_test_metadata.csv"


def getCsvFromHuggingFace(filename: str = "train.csv", repo_id: str = "osv5m/osv5m", repo_type: str = "dataset"):
    csv_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)
    df = pd.read_csv(csv_path)
    return df

def sample_train_and_val_split():
    osv_train_df = getCsvFromHuggingFace(filename=HF_TRAIN)
    train_quotas_df = pd.read_csv(TRAIN_QUOTA)
    val_quotas_df = pd.read_csv(VAL_QUOTA)

    assert train_quotas_df["country"].tolist() == val_quotas_df["country"].tolist(), "train/val quota ordering mismatch"

    # randomly shuffle the rows before sampling
    shuffled_df = osv_train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    sampled_train_rows = []
    sampled_val_rows = []
    for i in range(len(train_quotas_df)):
        country = train_quotas_df.iloc[i]["country"]
        val_country = val_quotas_df.iloc[i]["country"]
        train_quota = train_quotas_df.iloc[i]["quota"]
        val_quota = val_quotas_df.iloc[i]["quota"]
        country_subset_df = shuffled_df[shuffled_df["country"] == country]

        train_selected = country_subset_df.head(train_quota)
        val_selected = country_subset_df.iloc[train_quota: train_quota + val_quota]
        sampled_train_rows.append(train_selected)
        sampled_val_rows.append(val_selected)
        print(f"TRAIN -> Country: {country}, Quota: {train_quota}, Selected: {len(train_selected)}, Difference: {train_quota-len(train_selected)}")
        print(f"VAL -> Country: {val_country}, Quota: {val_quota}, Selected: {len(val_selected)}, Difference: {val_quota-len(val_selected)}")

    sampled_train_df = pd.concat(sampled_train_rows, ignore_index=True)
    sampled_val_df = pd.concat(sampled_val_rows, ignore_index=True)

    overlap = set(sampled_train_df["id"]).intersection(set(sampled_val_df["id"]))
    print(f"train/validation overlap: {len(overlap)}")
    assert len(overlap) == 0, "train/val overlap detected"

    sampled_train_df.to_csv(SAMPLED_TRAIN, index=False)
    sampled_val_df.to_csv(SAMPLED_VAL, index=False)


def sample_test_split():
    osv_test_df = getCsvFromHuggingFace(filename=HF_TEST)
    test_quotas_df = pd.read_csv(TEST_QUOTA)

    # randomly shuffle the rows before sampling
    shuffled_df = osv_test_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    sampled_rows = []
    for i in range(len(test_quotas_df)):
        country = test_quotas_df.iloc[i]["country"]
        quota = test_quotas_df.iloc[i]["quota"]
        country_subset_df = shuffled_df[shuffled_df["country"] == country]
        selected = country_subset_df.head(quota)
        sampled_rows.append(selected)
        print(f"Country: {country}, Quota: {quota}, Selected: {len(selected)}, Difference: {quota-len(selected)}")
        
    sampled_test_df = pd.concat(sampled_rows, ignore_index=True)
    sampled_test_df.to_csv(SAMPLED_TEST, index=False)

def main():
    print("Sampling train and validation splits from OSV-5M:")
    sample_train_and_val_split()
    print("Sampling test split from OSV-5M:")
    sample_test_split()

if __name__ == "__main__":
    main()