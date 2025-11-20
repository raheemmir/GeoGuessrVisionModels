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

test_set_csv = "dataset/metadata/processed/v2_h3_r2/test_metadata.csv"
id_to_h3_json = "dataset/metadata/processed/v2_h3_r2/id_to_h3.json"
h3_to_id_json = "dataset/metadata/processed/v2_h3_r2/h3_to_id.json"
output_dir = "evaluation/results"
root_dir = "." # for actually getting the images

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

def eval_performance(model_chkpt, model_name, is_resnet):
    df = pd.read_csv(test_set_csv)
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
    if is_resnet:
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        preprocess = weights.transforms()
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        model.load_state_dict(torch.load(model_chkpt, map_location="cpu"))
        model = model.to(device)
        model.eval()

    else: 
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
        h3_label = df.iloc[i]["h3_r2"] 
        img_path = df.iloc[i]["image_paths"]
        full_img_path = os.path.join(root_dir, img_path)
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
    models_to_evaluate = [
        {
            "model_chkpt": "model/resnet50/outputs/linear_probe_smoothing_wd/best_model_params.pt",
            "model_name": "resnet50_linear_probe_lr_1e-2_bs_64_step_8_reg",
            "is_resnet": True
        },
        {
            "model_chkpt": "model/resnet50/outputs/partial_fine_tune_smoothing_high_lr/best_model_params.pt",
            "model_name": "resnet50_layer4_ft_lr_1e-2_bs_64_step_8_reg",
            "is_resnet": True 
        },
        {
            "model_chkpt": "model/clip/outputs/clip_linear_probe_higher_lr/best_model_params.pt",
            "model_name": "clip_vitb32_linear_probe_lr_2e-2_bs_64_step_13_reg",
            "is_resnet": False
        },
        {
            "model_chkpt": "model/clip/outputs/clip_vitb32_linear_probe_lr_1e-3_bs_64_adamw_rlrop_epochs_40/best_model_params.pt",
            "model_name": "clip_vitb32_linear_probe_lr_1e-3_bs_64_adamw_rlrop_epochs_40",
            "is_resnet": False
        }

    ]

    model_metrics = []

    for config in models_to_evaluate:
        print("\n" + "=" * 80)
        print(f"Evaluating model on test set: {config["model_name"]}")
        print("=" * 80)

        df, metrics = eval_performance(
            model_chkpt=config["model_chkpt"], 
            model_name=config["model_name"],
            is_resnet=config["is_resnet"]    
        )

        predictions_csv = os.path.join(output_dir, f"{config["model_name"]}_test_predictions.csv")
        df.to_csv(predictions_csv, index=False)
        print(f"Saved per sample predictions to {predictions_csv}")
        model_metrics.append(metrics)

    metrics_df = pd.DataFrame(model_metrics)
    metrics_csv = os.path.join(output_dir, "model_comparison_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Saved metrics to {metrics_csv}")

if __name__ == "__main__":
    main()
