import torch
import torch.nn as nn

class ClipLinearProbe(nn.Module):
    def __init__(self, clip_model, num_classes):
        super().__init__()
        self.clip_model = clip_model
        for p in self.clip_model.parameters():
            p.requires_grad = False
        in_features = self.clip_model.visual.output_dim
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.clip_model.encode_image(x).float()
        logits = self.classifier(features)
        return logits           