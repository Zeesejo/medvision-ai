"""
Extra transform utilities — test-time augmentation (TTA) wrappers.
"""
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def tta_transforms(img_size: int = 224) -> list:
    """
    Returns a list of albumentations pipelines for test-time augmentation.
    Each pipeline is applied to the same image; predictions are averaged.
    """
    base = [
        A.Resize(img_size, img_size),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ]
    return [
        A.Compose(base),
        A.Compose([A.Resize(img_size, img_size), A.HorizontalFlip(p=1.0),
                   A.Normalize(mean=MEAN, std=STD), ToTensorV2()]),
        A.Compose([A.Resize(int(img_size * 1.1), int(img_size * 1.1)),
                   A.CenterCrop(img_size, img_size),
                   A.Normalize(mean=MEAN, std=STD), ToTensorV2()]),
    ]


def tta_predict(model, image_np: np.ndarray, transforms: list, device: str = "cuda") -> torch.Tensor:
    """
    Run TTA on a single numpy image [H, W, 3].
    Returns averaged sigmoid probabilities [num_classes].
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for tfm in transforms:
            img_t = tfm(image=image_np)["image"].unsqueeze(0).to(device)
            logits = model(img_t)
            preds.append(torch.sigmoid(logits).cpu())
    return torch.stack(preds).mean(dim=0).squeeze(0)
