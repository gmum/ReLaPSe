# swift/plugin/utils/metrics/object_eval.py

from __future__ import annotations

from typing import Tuple
import torch
from PIL import Image
from transformers import AutoImageProcessor, ResNetForImageClassification


@torch.no_grad()
def imagenet_ResNet50(device: torch.device | str):
    """
    Returns (processor, classifier) where classifier outputs 1000 ImageNet-1k logits.
    """
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    classifier = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
    classifier.to(device)
    classifier.eval()
    classifier.requires_grad_(False)
    return processor, classifier


@torch.no_grad()
def object_eval(
    classifier: ResNetForImageClassification,
    image: Image.Image,
    *,
    processor: AutoImageProcessor,
    device: torch.device | str,
) -> Tuple[int, torch.Tensor]:
    """
    Returns:
      pred_idx: int (ImageNet class index 0..999)
      logits: torch.Tensor shape [1000] (pre-softmax)
    """
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = classifier(**inputs)
    logits = out.logits[0]  # [1000]
    pred_idx = int(torch.argmax(logits).item())
    return pred_idx, logits
