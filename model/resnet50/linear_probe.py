import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.model_selection import train_test_split
import os
from model.resnet50.dataset import StreetViewImageDataset
import time

def train_linear_probe(input_csv, root_dir, out_dir, batch_size, num_workers, base_lr):
    os.makedirs(out_dir, exist_ok=True)
    input_df = pd.read_csv(input_csv)
    num_classes = input_df["class_id"].max() + 1
    train_df, val_df = train_test_split(
        input_df, 
        test_size=0.05, 
        random_state=42
    )
    splits_dir = os.path.join(out_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)
    train_csv = os.path.join(splits_dir, "train_split.csv")
    val_csv = os.path.join(splits_dir, "val_split.csv")
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    weights = models.ResNet50_Weights.IMAGENET1K_V2
    resnet50_transforms = weights.transforms()

    train_dataset = StreetViewImageDataset(csv=train_csv, root_dir=root_dir, transform=resnet50_transforms)
    val_dataset = StreetViewImageDataset(csv=val_csv, root_dir=root_dir, transform=resnet50_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    dataloaders = {"train": train_dataloader, "val": val_dataloader}
    dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}

    #device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = models.resnet50(weights=weights)
    for p in model.parameters():
        p.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=base_lr, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

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
        num_epochs=25,
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
        num_epochs=25, 
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
                running_corrects += (preds == labels).sum().item() # torch.sum(preds == labels)
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            with open(metrics_path, "a") as f:
                f.write(f"{epoch},{phase},{epoch_loss},{epoch_acc}\n")
            
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_path)
            
        torch.save(model.state_dict(), last_path) # save every epoch
        print(f'Saved checkpoint to {last_path}')

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(torch.load(best_path))
    return model