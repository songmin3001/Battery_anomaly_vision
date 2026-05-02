"""
model.py
EfficientNet-B3 기반 배터리 불량 분류 모델
"""

import torch
import torch.nn as nn
from torchvision import models
from dataset import NUM_CLASSES


class BatteryClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.4):
        super().__init__()

        backbone = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1
        )

        self.features = backbone.features
        self.avgpool = backbone.avgpool
        in_features = backbone.classifier[1].in_features  # 1536

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.SiLU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def predict_with_score(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)

        normal_prob = probs[:, 0]
        anomaly_score = 1.0 - normal_prob
        pred_class = probs.argmax(dim=-1)
        return pred_class, anomaly_score, probs


def build_model(device="cuda"):
    model = BatteryClassifier()
    return model.to(device)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"전체 파라미터: {total:,} | 학습 가능: {trainable:,}"
