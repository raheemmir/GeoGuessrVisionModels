import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import clip
import os
from model.resnet50.dataset import StreetViewImageDataset
from model.clip.clip_wrapper import ClipLinearProbe
import time
        
def clip_linear_probe(train_csv, val_csv, id_col, root_dir, out_dir, batch_size, num_workers, base_lr):
    os.makedirs(out_dir, exist_ok=True)
    train_df = pd.read_csv(train_csv)
    num_classes = train_df[id_col].max() + 1 

    device = torch.device('xpu' if torch.xpu.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'xpu':
        print(f"Intel GPU: {torch.xpu.get_device_name(0)}")

    clip_model, clip_preprocess = clip.load('ViT-B/32', device=device, jit=False)
    clip_model = clip_model.float() # casts everything to fp32

    train_dataset = StreetViewImageDataset(csv=train_csv, root_dir=root_dir, labels_col=id_col, transform=clip_preprocess)
    val_dataset = StreetViewImageDataset(csv=val_csv, root_dir=root_dir, labels_col=id_col, transform=clip_preprocess)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    dataloaders = {"train": train_dataloader, "val": val_dataloader}
    dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}
    
    model = ClipLinearProbe(clip_model=clip_model, num_classes=num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.classifier.parameters(), lr=base_lr, weight_decay=0.01)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="max", factor=0.5, patience=3, threshold=0.001, threshold_mode="abs", min_lr=1e-6)

    metrics_path = os.path.join(out_dir, "metrics.csv")
    best_path = os.path.join(out_dir, "best_model_params.pt")
    last_path = os.path.join(out_dir, "last_model_params.pt")
    with open(metrics_path, "w") as f:
        f.write("epoch,phase,loss,acc\n")

    model = train_model(
        model=model, 
        dataloaders=dataloaders, 
        dataset_sizes=dataset_sizes,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=40,
        metrics_path=metrics_path,
        best_path=best_path,
        last_path=last_path
    )

    return model, {"best": best_path, "last": last_path, "metrics": metrics_path}


def train_model(
        model, 
        dataloaders, 
        dataset_sizes,
        criterion, 
        optimizer, 
        scheduler, 
        device,
        num_epochs=40, 
        metrics_path=None, 
        best_path=None,
        last_path=None
):
    since = time.time()

    torch.save(model.state_dict(), best_path)
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = outputs.argmax(dim=1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum().item()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            with open(metrics_path, "a") as f:
                f.write(f"{epoch},{phase},{epoch_loss},{epoch_acc}\n")
            
            if phase == "val":
                scheduler.step(epoch_acc)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_path)
            
        torch.save(model.state_dict(), last_path) # save every epoch
        print(f'Saved checkpoint to {last_path}')

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(torch.load(best_path))
    return model


def main():
    TRAIN_CSV_R1 = "v2/assets/train_metadata_h3_r1_with_ids.csv"
    TRAIN_CSV_R2 = "v2/assets/train_metadata_h3_r2_with_ids.csv"
    VAL_CSV_R1 = "v2/assets/val_metadata_h3_r1_with_ids.csv"
    VAL_CSV_R2 = "v2/assets/val_metadata_h3_r2_with_ids.csv"
    ROOT_DIR = "."
    H3_RESOLUTION = 2
    H3_ID_COL = f"id_h3_r{H3_RESOLUTION}"
    OUT_DIR = f"v2/training/outputs/clip_vitb32_linear_probe_r{H3_RESOLUTION}"
    BATCH_SIZE = 64
    NUM_WORKERS = 8
    BASE_LR = 0.001

    model, paths = clip_linear_probe(
        train_csv=TRAIN_CSV_R2,
        val_csv=VAL_CSV_R2,
        id_col=H3_ID_COL,
        root_dir=ROOT_DIR,
        out_dir=OUT_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        base_lr=BASE_LR
    )
    print(paths)

if __name__ == "__main__":
    main()