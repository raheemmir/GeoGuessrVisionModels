# Using country distribution data computed for each split from v1
# filtering and recomputing distribution data based on new country list (geoguessr countries)

import pandas as pd

OSV_TRAIN_DISTRIBUTIONS = "dataset/country_distributions/train_split_country_distributions.csv"
OSV_TEST_DISTRIBUTIONS = "dataset/country_distributions/test_split_country_distributions.csv"
COUNTRY_LIST_IN = "v2/assets/geoguessr_countries_iso2.txt"
COUNTRY_LIST_OUT_OSV = "v2/assets/geoguessr_countries_filtered_osv.txt"
COUNTRY_LIST_OUT_FINAL = "v2/assets/geoguessr_countries_filtered_final.txt"
TRAIN_DISTRIBUTIONS_OUT = "v2/assets/train_split_country_distributions_filtered.csv"
TEST_DISTRIBUTIONS_OUT = "v2/assets/test_split_country_distributions_filtered.csv"
MIN_COUNT = 1000

def filter_country_list_to_osv():
    osv_train_df = pd.read_csv(OSV_TRAIN_DISTRIBUTIONS)
    osv_train_countries = osv_train_df["country"].to_list()
    osv_test_df = pd.read_csv(OSV_TEST_DISTRIBUTIONS)
    osv_test_countries = osv_test_df["country"].to_list()

    with open(COUNTRY_LIST_IN, "r") as f:
        country_list = [line.strip() for line in f if line.strip()]

    missing_in_train = []
    missing_in_test = []
    missing_countries = set()

    print("Countries missing from OSV-5M train:")
    for c in country_list:
        if c not in osv_train_countries:
            missing_in_train.append(c)
            missing_countries.add(c)
            print(c)
    
    print("Countries missing from OSV-5M test:")
    for c in country_list:
        if c not in osv_test_countries:
            missing_in_test.append(c)
            missing_countries.add(c)
            print(c)

    print("Countries missing from OSV-5M (overall):")
    for c in missing_countries:
        print(c)
    
    filtered_country_list = [c for c in country_list if c not in missing_countries]

    print(f"Length of country list before: {len(country_list)}")
    print(f"Length of country list after OSV-5M filtering: {len(filtered_country_list)}")

    with open(COUNTRY_LIST_OUT_OSV, "w") as f:
        for c in filtered_country_list:
            f.write(c + "\n")

    print("Wrote intermediate country list to: " + COUNTRY_LIST_OUT_OSV)

def build_final_country_list():
    with open(COUNTRY_LIST_OUT_OSV, "r") as f:
        country_list = [line.strip() for line in f if line.strip()]

    train_df = pd.read_csv(OSV_TRAIN_DISTRIBUTIONS)
    train_df.drop(columns=["proportion", "percentage"], inplace=True)

    rows_to_drop = []
    for idx, row in train_df.iterrows():
        country = row["country"]
        count = row["count"]
        if country not in country_list or count < MIN_COUNT:
            rows_to_drop.append(idx)
    train_df.drop(index=rows_to_drop, inplace=True)
    train_df.reset_index(drop=True, inplace=True)
    train_df.sort_values("count", ascending=False, inplace=True)

    final_countries = train_df["country"].to_list()

    with open(COUNTRY_LIST_OUT_FINAL, "w") as f:
        for c in final_countries:
            f.write(c + "\n")
    print(f"Length of country list after filtering out countries w/ count < {MIN_COUNT}: {len(final_countries)}")
    print("Wrote final country list to: " + COUNTRY_LIST_OUT_FINAL)


def update_country_distributions():
    train_df = pd.read_csv(OSV_TRAIN_DISTRIBUTIONS)
    test_df = pd.read_csv(OSV_TEST_DISTRIBUTIONS)
    dfs = [train_df, test_df]

    with open(COUNTRY_LIST_OUT_FINAL, "r") as f:
        country_list = [line.strip() for line in f if line.strip()]

    for df in dfs:
        # remove old columns + filter countries
        df.drop(columns=["proportion", "percentage"], inplace=True)
        rows_to_drop = []
        for idx, row in df.iterrows():
            country = row["country"]
            if country not in country_list:
                rows_to_drop.append(idx)
        df.drop(index=rows_to_drop, inplace=True)
        df.reset_index(drop=True, inplace=True)

        # recompute country distributions
        total = df["count"].sum()
        df["proportion"] = df["count"] / total
        df["percentage"] = (df["proportion"] * 100).round(3)

        df.sort_values("count", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
    
    train_df.to_csv(TRAIN_DISTRIBUTIONS_OUT, index=False)
    print("Wrote updated train-split country distributions to: " + TRAIN_DISTRIBUTIONS_OUT)
    test_df.to_csv(TEST_DISTRIBUTIONS_OUT, index=False)
    print("Wrote updated test-split country distributions to: " + TEST_DISTRIBUTIONS_OUT)

def main():
    filter_country_list_to_osv()
    build_final_country_list()
    update_country_distributions()
    
if __name__ == "__main__":
    main()