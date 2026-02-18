import pandas as pd
import torch
import torch.nn as nn
from torchvision import models
import clip
from model.clip.clip_wrapper import ClipLinearProbe
from PIL import Image
import h3
import os
import json

ROOT_DIR = "." # for actually getting the images
OUTPUT_DIR = "v2/evaluation/results"

# from https://stackoverflow.com/a/4913653
from math import radians, cos, sin, asin, sqrt

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r

def eval_performance(resolution, model_name, model_chkpt, test_csv, id_to_h3_json, h3_to_id_json):
    df = pd.read_csv(test_csv)
    with open(id_to_h3_json, "r") as f:
        id_to_h3_raw = json.load(f)
        id_to_h3 = {int(k): v for k, v in id_to_h3_raw.items()} # casting key to int
    with open(h3_to_id_json, "r") as f:
        h3_to_id = json.load(f)
    
    num_classes = len(h3_to_id)
    num_samples = len(df)

    device = torch.device('xpu' if torch.xpu.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'xpu':
        print(f"Intel GPU: {torch.xpu.get_device_name(0)}")

    # building model archicture(s) and loading weights
    clip_model, preprocess = clip.load('ViT-B/32', device="cpu", jit=False)
    clip_model = clip_model.float() # casts everything to fp32
    clip_model.eval()
    model = ClipLinearProbe(clip_model, num_classes)
    model.load_state_dict(torch.load(model_chkpt, map_location="cpu"))
    model = model.to(device)
    model.eval()

    # evaluating model on the test set
    pred_class_ids = []
    pred_h3_cells = []
    pred_lats = []
    pred_lons = []
    distance_errors_km = []
    correct_preds = []
    seen_in_train = []
    unseen_in_train_count = 0 # number of test samples with h3 cells not seen in training

    for i in range(num_samples):
        h3_label = df.iloc[i][f"h3_r{resolution}"] 
        img_path = df.iloc[i]["image_paths"]
        full_img_path = os.path.join(ROOT_DIR, img_path)
        img = Image.open(full_img_path).convert("RGB")
        img = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(img)
            pred = logits.argmax(dim=1)
            pred_id = int(pred.item())
        
        pred_class_ids.append(pred_id)
        pred_h3 = id_to_h3[pred_id]
        pred_h3_cells.append(pred_h3)
        pred_lat, pred_lon = h3.cell_to_latlng(pred_h3) 
        pred_lats.append(pred_lat)
        pred_lons.append(pred_lon)
        true_lat = df.iloc[i]["latitude"]
        true_lon = df.iloc[i]["longitude"]
        error_km = haversine(lon1=true_lon, lat1=true_lat, lon2=pred_lon, lat2=pred_lat)
        distance_errors_km.append(error_km)

        if h3_label not in h3_to_id:
            correct = False
            from_train_set = False
            unseen_in_train_count += 1
        else:
            correct = (h3_label == pred_h3)
            from_train_set = True
        
        correct_preds.append(correct)
        seen_in_train.append(from_train_set)

    df["pred_class_id"] = pred_class_ids
    df["pred_h3"] = pred_h3_cells
    df["pred_lat"] = pred_lats
    df["pred_lon"] = pred_lons
    df["error_km"] = distance_errors_km 
    df["is_correct"] = correct_preds 
    df["seen_in_train"] = seen_in_train  

    # metrics
    mean_error_km = df["error_km"].mean()
    q10 = df["error_km"].quantile(0.1)
    q25 = df["error_km"].quantile(0.25)
    q50 = df["error_km"].quantile(0.5) # median
    q75 = df["error_km"].quantile(0.75)
    q90 = df["error_km"].quantile(0.9)
    q95 = df["error_km"].quantile(0.95)
    q99 = df["error_km"].quantile(0.99)
    overall_acc = df["is_correct"].mean()
    filtered_df = df[df["seen_in_train"]]
    seen_acc = filtered_df["is_correct"].mean()
    prop_of_samples_with_unseen_labels = unseen_in_train_count / num_samples

    metrics = {
        "model_name": model_name,
        "mean_error_km": mean_error_km,
        "q10_error_km": q10,
        "q25_error_km": q25,
        "q50_error_km": q50,
        "q75_error_km": q75,
        "q90_error_km": q90,
        "q95_error_km": q95,
        "q99_error_km": q99,
        "overall_acc": overall_acc,
        "seen_acc": seen_acc,
        "prop_unseen_labels": prop_of_samples_with_unseen_labels
    }

    return df, metrics

def main():
    eval_configs = {
        1: {
            "model_name": "clip_vitb32_linear_probe_r1",
            "model_chkpt": "v2/training/outputs/clip_vitb32_linear_probe_r1/best_model_params.pt",
            "test_csv": "v2/assets/test_metadata_h3_r1.csv",
            "id_to_h3_json": "v2/assets/id_to_h3_r1.json",
            "h3_to_id_json": "v2/assets/h3_r1_to_id.json"
        },
        2: {
            "model_name": "clip_vitb32_linear_probe_r2",
            "model_chkpt": "v2/training/outputs/clip_vitb32_linear_probe_r2/best_model_params.pt",
            "test_csv": "v2/assets/test_metadata_h3_r2.csv",
            "id_to_h3_json": "v2/assets/id_to_h3_r2.json",
            "h3_to_id_json": "v2/assets/h3_r2_to_id.json"
        }
    }

    model_metrics = []

    for res, config in eval_configs.items():
        print("\n" + "=" * 80)
        print(f"H3 Resolution {res}: ")
        print(f"Evaluating {config["model_name"]} on {config["test_csv"]}")

        predictions_df, metrics = eval_performance(
            resolution=res,
            model_name=config["model_name"],
            model_chkpt=config["model_chkpt"],
            test_csv=config["test_csv"],
            id_to_h3_json=config["id_to_h3_json"],
            h3_to_id_json=config["h3_to_id_json"]
        )

        predictions_csv = os.path.join(OUTPUT_DIR, f"{config["model_name"]}_test_predictions.csv")
        predictions_df.to_csv(predictions_csv, index=False)
        print(f"Saved per sample predictions to {predictions_csv}")
        model_metrics.append(metrics)

    metrics_df = pd.DataFrame(model_metrics)
    metrics_csv = os.path.join(OUTPUT_DIR, "fine_tuned_clip_models_test_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Saved metrics to {metrics_csv}")

if __name__ == "__main__":
    main()
