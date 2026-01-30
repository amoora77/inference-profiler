import torch
from torchvision import transforms


def get_vision_preprocessor():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def create_vision_input(batch_size, device, channels_last=False):
    img = torch.randn(batch_size, 3, 224, 224, device=device)
    if channels_last:
        img = img.to(memory_format=torch.channels_last)
    return img


def preprocess_vision_batch(batch_size, device, channels_last=False):
    raw = torch.randn(batch_size, 3, 256, 256, device=device)
    preprocessor = get_vision_preprocessor()
    processed = preprocessor(raw)
    if channels_last:
        processed = processed.to(memory_format=torch.channels_last)
    return processed


def create_text_input(batch_size, seq_len=128, vocab_size=10000, device="cpu"):
    return torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
