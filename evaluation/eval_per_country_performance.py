import pandas as pd
import os

def eval_per_country_performance(test_predictions_csv, model_name, out_dir):
    print(f"Analyzing per-country test set performance for: {model_name}")
    df = pd.read_csv(test_predictions_csv)
    countries = df["country"].unique().tolist()
    per_country_metrics = []
    for country in countries:
        country_df = df[df["country"] == country]
        num_samples = len(country_df)
        mean_error_km = country_df["error_km"].mean()
        q10 = country_df["error_km"].quantile(0.1)
        q25 = country_df["error_km"].quantile(0.25)
        q50 = country_df["error_km"].quantile(0.5) # median
        q75 = country_df["error_km"].quantile(0.75)
        q90 = country_df["error_km"].quantile(0.9)
        q95 = country_df["error_km"].quantile(0.95)
        q99 = country_df["error_km"].quantile(0.99)
        overall_acc = country_df["is_correct"].mean()
        seen_df = country_df[country_df["seen_in_train"]]
        seen_acc = seen_df["is_correct"].mean() if len(seen_df) > 0 else float("nan")

        per_country_metrics.append({
            "country": country,
            "num_samples": num_samples,
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
        }) 

    country_metrics_df = pd.DataFrame(per_country_metrics).sort_values("num_samples", ascending=False).reset_index(drop=True)
    top25_country_metrics_df = country_metrics_df.head(25) # represents ~60% of dataset

    per_country_metrics_csv = os.path.join(out_dir, f"{model_name}_per_country_metrics.csv")
    country_metrics_df.to_csv(per_country_metrics_csv, index=False)
    print(f"Saved: {per_country_metrics_csv}")
    top25_per_country_metrics_csv = os.path.join(out_dir, f"{model_name}_top_25_per_country_metrics.csv")
    top25_country_metrics_df.to_csv(top25_per_country_metrics_csv, index=False)
    print(f"Saved: {top25_per_country_metrics_csv}")

def main():

    models_to_evaluate = [
        {
            "model_name": "resnet50_linear_probe_lr_1e-2_bs_64_step_8_reg",
            "test_predictions_csv": "evaluation/results/resnet50_linear_probe_lr_1e-2_bs_64_step_8_reg_test_predictions.csv"
        },
        {
            "model_name": "resnet50_layer4_ft_lr_1e-2_bs_64_step_8_reg",
            "test_predictions_csv": "evaluation/results/resnet50_layer4_ft_lr_1e-2_bs_64_step_8_reg_test_predictions.csv"
        },
        {
            "model_name": "clip_vitb32_linear_probe_lr_2e-2_bs_64_step_13_reg",
            "test_predictions_csv": "evaluation/results/clip_vitb32_linear_probe_lr_2e-2_bs_64_step_13_reg_test_predictions.csv"
        },
        {
            "model_name": "clip_vitb32_linear_probe_lr_1e-3_bs_64_adamw_rlrop_epochs_40",
            "test_predictions_csv": "evaluation/results/clip_vitb32_linear_probe_lr_1e-3_bs_64_adamw_rlrop_epochs_40_test_predictions.csv"
        }
    ] 

    for config in models_to_evaluate:
        print("\n" + "=" * 80)
        print(f"Analyzing per-country test set performance for: {config["model_name"]}")
        print("=" * 80)

        output_dir = "evaluation/results/per_country"
        os.makedirs(output_dir, exist_ok=True)

        eval_per_country_performance(config["test_predictions_csv"], config["model_name"], output_dir)


if __name__ == "__main__":
    main()
