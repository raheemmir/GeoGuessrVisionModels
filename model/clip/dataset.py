import pandas as pd
import os
from torch.utils.data import Dataset
from PIL import Image

class StreetViewImageDataset(Dataset):
    def __init__(self, csv, root_dir, img_paths_col="image_paths", labels_col="class_id", transform=None):
        df = pd.read_csv(csv)
        self.root_dir = root_dir
        self.image_paths = df[img_paths_col].astype(str).to_list()
        self.labels = df[labels_col].astype(int).to_list()
        self.transform = transform


        # Test first path in case
        print(f"Root dir: {self.root_dir}")
        print(f"Sample path from CSV: {self.image_paths[0]}")
        full_path = os.path.join(self.root_dir, self.image_paths[0])
        print(f"Full path: {full_path}")
        print(f"Exists: {os.path.isfile(full_path)}")

        # check if any are missing
        missing_paths = [p for p in self.image_paths if not os.path.isfile(os.path.join(self.root_dir, p))]
        if missing_paths:
            print(f"Warning: Missing {len(missing_paths)} images!")

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_paths[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label