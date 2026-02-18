import pandas as pd
import os
import shutil
import json

PREDICTIONS_CSV = "v2/evaluation/results/clip_vitb32_linear_probe_r2_test_predictions.csv"
ROOT_DIR = "." # for retrieving images
OUTPUT_IMAGES_DIR = "v2/evaluation/demo_samples/images"
OUTPUT_PARENT_DIR = "v2/evaluation/demo_samples"

def select_and_export_demo_samples():
    os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
    select_countries = ["US", "DE", "FR", "RU", "JP", "AU", "BR", "GB", "SE", "IT"]
    df = pd.read_csv(PREDICTIONS_CSV)
    df_sorted = df.sort_values("error_km", ascending=True)
    df_filtered = df_sorted[df_sorted['country'].isin(select_countries)]

    top1000_df = df_filtered.head(1000)
    top25_df = top1000_df.sample(n=25)
    top25_df = top25_df.sort_values("error_km", ascending=True).reset_index(drop=True)

    demo_predictions_csv = os.path.join(OUTPUT_PARENT_DIR, "demo_top25_r2_predictions.csv")
    top25_df.to_csv(demo_predictions_csv, index=False)
    print(f"Saved top 25 predictions (resolution 2) to {demo_predictions_csv}")

    demo_samples = []

    for i in range(len(top25_df)):
        relative_path = top25_df.iloc[i]["image_paths"]
        filename = os.path.basename(relative_path)
        src_path = os.path.join(ROOT_DIR, relative_path)
        dest_path = os.path.join(OUTPUT_IMAGES_DIR, filename)

        shutil.copy2(src_path, dest_path)

        demo_samples.append({
            "id": i+1,
            "path": f"/images/{filename}",
            "groundTruthLat": top25_df.iloc[i]["latitude"],
            "groundTruthLng": top25_df.iloc[i]["longitude"],
            "clipPredLat": top25_df.iloc[i]["pred_lat"],
            "clipPredLng": top25_df.iloc[i]["pred_lon"],
            "error_km": top25_df.iloc[i]["error_km"],
            "country": top25_df.iloc[i]["country"],
            "region": top25_df.iloc[i]["region"],
            "city": top25_df.iloc[i]["city"]
        })
    
    demo_predictions_json = os.path.join(OUTPUT_PARENT_DIR, "demo_predictions.json")
    with open(demo_predictions_json, "w") as f:
        json.dump(demo_samples, f, indent=2)

    print(f"Copied {len(demo_samples)} images to {OUTPUT_IMAGES_DIR}")
    print(f"Saved demo predictions JSON to  {demo_predictions_json}")
    

def main():
    select_and_export_demo_samples()

if __name__ == "__main__":
    main()