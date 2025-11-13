from huggingface_hub import hf_hub_download
import pandas as pd

def analyze_dataset(filename="train.csv", repo_id="osv5m/osv5m", repo_type="dataset"):
    csv_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)
    df = pd.read_csv(csv_path, usecols=["id", "latitude", "longitude", "country"])
    total_samples = df.shape[0]

    print(f"Dataframe shape: {df.shape}")
    print(f"Dataframe columns: {list(df.columns)}")

    country_totals = {}
    for country in df["country"]:
        if country not in country_totals:
            country_totals[country] = 1
        else:
            country_totals[country] += 1

    print(f"Total countries: {len(country_totals.keys())}")
    print(f"United states total occurences: {country_totals['US']}") # sanity check should ~25% of dataset

    country_totals_df = pd.DataFrame(list(country_totals.items()), columns=["country", "count"]).sort_values("count", ascending=False).reset_index(drop=True)
    country_totals_df["proportion"] = country_totals_df["count"] / total_samples
    country_totals_df["percentage"] = (country_totals_df["proportion"] * 100).round(3)
    print("Country Totals Dataframe:")
    print(country_totals_df.head(25))
    print(f"Sum of proportion column (should be 1.0): {country_totals_df['proportion'].sum()}")
    top_prop = 22
    print(f"Total proportion of the top {top_prop}: {country_totals_df.head(top_prop)['proportion'].sum()}")
    bottom_prop = 200
    print(f"Total of the bottom {bottom_prop}: {country_totals_df.tail(bottom_prop)['proportion'].sum()}")

    file = filename.split(".")[0]
    country_totals_df.to_csv(f"dataset/country_distributions/{file}_split_country_distributions.csv", index=False)

def main(): 
    print("Analyzing osv5m: train.csv")
    analyze_dataset()
    print("Analyzing osv5m: test.csv")
    analyze_dataset(filename="test.csv")

if __name__ == "__main__":
    main()
