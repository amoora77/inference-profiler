import torch
import torch.nn as nn
from torchvision import models


class TinyTransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size=10000,
        embed_dim=256,
        num_heads=4,
        num_layers=4,
        seq_len=128,
        num_classes=10,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = x + self.pos_embedding[:, : input_ids.size(1), :]
        x = self.transformer(x)
        pooled = x.mean(dim=1)
        logits = self.classifier(pooled)
        return logits


def get_model(name, device, quantize=False, channels_last=False):
    model = None

    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    elif name == "tiny_transformer":
        model = TinyTransformerEncoder()
    else:
        raise ValueError(f"Unknown model: {name}")

    model.eval()
    model.to(device)

    if channels_last and name in ["resnet18", "mobilenet_v3_small"]:
        model = model.to(memory_format=torch.channels_last)

    if quantize and name == "tiny_transformer":
        model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )

    return model


def is_vision_model(name):
    return name in ["resnet18", "mobilenet_v3_small"]


def is_text_model(name):
    return name == "tiny_transformer"
