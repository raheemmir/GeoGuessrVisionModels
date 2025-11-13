from huggingface_hub import hf_hub_download
import pandas as pd

SEED = 42
OUTPUT_DIR = "dataset/dataset_construction"

def getCsvFromHuggingFace(filename: str = "train.csv", repo_id: str = "osv5m/osv5m", repo_type: str = "dataset"):
    csv_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)
    df = pd.read_csv(csv_path)
    return df

def sample_dataset(hugging_face_filename: str, quota_csv: str, output_filename: str):
    dataset_df = getCsvFromHuggingFace(hugging_face_filename)
    quotas_df = pd.read_csv(quota_csv)

    # randomly shuffle the rows before sampling
    shuffled_df = dataset_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    sampled_rows = []
    logs = []
    for i in range(len(quotas_df)):
        country = quotas_df.iloc[i]["country"]
        quota = quotas_df.iloc[i]["quota_ceil"]
        country_subset_df = shuffled_df[shuffled_df["country"] == country]
        selected = country_subset_df.head(quota)
        sampled_rows.append(selected)
        log = {"country": country, "quota": quota, "achieved": len(selected), "difference": quota-len(selected)}
        print(log)
        logs.append(log)

    sampled_dataset_df = pd.concat(sampled_rows, ignore_index=True)

    # some verification checks 
    print(f"Sum of quotas: {quotas_df['quota_ceil'].sum()}")
    print(f"Number of samples selected: {len(sampled_dataset_df)}")
    print(f"Number of unique ids (there should be {quotas_df['quota_ceil'].sum()}): {sampled_dataset_df['id'].nunique()}")
    print(f"{hugging_face_filename}'s columns: {dataset_df.columns}")
    print(f"Sampled dataset's columns: {sampled_dataset_df.columns}")
    achieved_country_counts_df = sampled_dataset_df['country'].value_counts().rename("achieved").reset_index().rename(columns={"index": "country"})
    quotas_copy_df = quotas_df.copy()
    comparison_df = quotas_copy_df.merge(achieved_country_counts_df, how="inner", on="country")
    comparison_df["diffs"] = comparison_df["quota_ceil"] - comparison_df["achieved"]
    print(f"Sum of differences between quotas and what was actually achieved (should be 0): {comparison_df['diffs'].sum()}")  

    dataset_output_path = OUTPUT_DIR + "/" + output_filename
    sampled_dataset_df.to_csv(dataset_output_path, index=False)

    logs_df = pd.DataFrame(logs).sort_values("quota", ascending=False)
    logs_output_filename = "logs_from_" + output_filename
    logs_output_path = OUTPUT_DIR + "/" + logs_output_filename 
    logs_df.to_csv(logs_output_path, index=False)


def main():
    sample_dataset(hugging_face_filename="train.csv", quota_csv="dataset/dataset_construction/train_quotas.csv", output_filename="sampled_train_dataset.csv")
    sample_dataset(hugging_face_filename="test.csv", quota_csv="dataset/dataset_construction/test_quotas.csv", output_filename="sampled_test_dataset.csv")

if __name__ == "__main__":
    main()