import pandas as pd
import matplotlib.pyplot as plt
import os


csv_paths = {
    "resent50_linear_probe": "model/resnet50/outputs/linear_probe_smoothing_wd/metrics.csv",
    "resnet50_last_layer_fine_tune": "model/resnet50/outputs/partial_fine_tune_smoothing_high_lr/metrics.csv",
    "clip_vitb32_linear_probe_A": "model/clip/outputs/clip_linear_probe_higher_lr/metrics.csv",
    "clip_vitb32_linear_probe_B": "model/clip/outputs/clip_vitb32_linear_probe_lr_1e-3_bs_64_adamw_rlrop_epochs_40/metrics.csv"
}

out_dir = "evaluation/plots"

def plot_training(metric="acc"):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    for ax, (model_name, csv_path) in zip(axes, csv_paths.items()):
        df = pd.read_csv(csv_path)
        train_df = df[df["phase"] == "train"]
        val_df = df[df["phase"] == "val"]

        ax.plot(train_df["epoch"], train_df[metric], label="train {metric}", linewidth=2)
        ax.plot(val_df["epoch"],   val_df[metric], label=f"val {metric}", linestyle="--", linewidth=2)
        ax.set_title(model_name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"training_curves_{metric}.png"), dpi=300)
    plt.show()

def main():
    plot_training("loss")

if __name__ == "__main__":
    main()
