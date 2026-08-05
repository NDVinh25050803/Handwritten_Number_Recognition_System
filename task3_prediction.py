"""
Task 3 — Character Recognition and Expression Evaluation (HNRS)

Classifies character segments produced by Task 2 using a trained PyTorch model.
Assembles per-character predictions into a left-to-right string.
Evaluates the string mathematically if it consists solely of valid arithmetic.
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field, replace
from typing import Literal

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

import task1_preprocessing as t1
import task2_segmentation as t2

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class PredictionConfig:
    """Prediction pipeline parameters."""

    model_type: Literal["cnn", "mobilenet_v3_small", "resnet18"] = "cnn"
    model_path: str = "best_cnn_model.pth"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    target_size: int = 28
    in_channels: int = 1

    class_map: list[str] = field(default_factory=lambda: [
        "(", ")", "*", "+", "-", "/",
        *list("0123456789"),
        *list("abcdefghijklmnopqrstuvwxyz")
    ])


DEFAULT_CONFIG = PredictionConfig()

_SYMBOL_TO_PYTHON = {"/": "/", "*": "*"}


class CustomCNN(nn.Module):
    """Custom CNN architecture for grayscale inputs."""

    def __init__(self, num_classes=42):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.dropout2d = nn.Dropout2d(p=0.25)

        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dropout2d(x)
        x = torch.flatten(x, 1)
        return self.fc_layers(x)


def load_model(config: PredictionConfig) -> nn.Module:
    """Initializes the specified architecture and loads trained weights."""
    checkpoint = torch.load(
        config.model_path, map_location=torch.device(config.device), weights_only=False
    )
    
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        extracted_class_map = checkpoint.get("class_names", config.class_map)
    else:
        state_dict = checkpoint
        extracted_class_map = config.class_map

    num_classes = len(extracted_class_map)

    if config.model_type == "cnn":
        model = CustomCNN(num_classes=num_classes)
        model.in_channels = 1
        model.target_size = 28
    elif config.model_type == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(model.classifier[3].in_features, num_classes)
        )
        model.in_channels = 3
        model.target_size = 224
    elif config.model_type == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(model.fc.in_features, num_classes)
        )
        model.in_channels = 3
        model.target_size = 224
    else:
        raise ValueError(f"Unsupported model_type: {config.model_type}")

    model.load_state_dict(state_dict)
    model.to(config.device)
    model.eval()
    
    model.class_map = extracted_class_map
    
    return model


def _predict_characters(
    crops: list[np.ndarray],
    model: nn.Module,
    config: PredictionConfig,
) -> str:
    """Executes batched inference on resized character segments."""
    in_channels = getattr(model, "in_channels", config.in_channels)
    class_map = getattr(model, "class_map", config.class_map)
    target_size = getattr(model, "target_size", config.target_size)

    processed_crops = []
    for crop in crops:
        if crop.shape != (target_size, target_size):
            crop = cv2.resize(
                crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR
            )
        processed_crops.append(crop)

    batch = np.stack(processed_crops).astype(np.float32) / 255.0
    
    if in_channels == 3:
        batch = np.repeat(batch[:, np.newaxis, :, :], 3, axis=1)
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]
    else:
        batch = batch[:, np.newaxis, :, :]
        mean_vals = [0.1532]
        std_vals = [0.3114]

    batch_tensor = torch.from_numpy(batch).to(config.device)
    
    mean_tensor = torch.tensor(mean_vals, device=config.device).view(1, -1, 1, 1)
    std_tensor = torch.tensor(std_vals, device=config.device).view(1, -1, 1, 1)
    batch_tensor = (batch_tensor - mean_tensor) / std_tensor

    with torch.no_grad():
        logits = model(batch_tensor)
        predicted = torch.argmax(logits, dim=1).cpu().numpy()

    return "".join(
        class_map[idx] if idx < len(class_map) else "?"
        for idx in predicted
    )


def _to_python_expression(expression: str) -> str:
    """Translates recognized operator symbols into standard Python syntax."""
    for symbol, python_op in _SYMBOL_TO_PYTHON.items():
        expression = expression.replace(symbol, python_op)
    return expression


def _format_result(expression: str) -> str:
    """Evaluates strings containing only digits and arithmetic operators."""
    if re.search(r"[a-zA-Z]", expression):
        return expression

    python_expression = _to_python_expression(expression)
    try:
        parsed = ast.parse(python_expression, mode="eval")
        result = eval(compile(parsed, filename="<string>", mode="eval"))
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return f"{expression} = {result}"
    except Exception:
        return expression


def predict_image(
    img_path: str,
    model: nn.Module,
    config: PredictionConfig = DEFAULT_CONFIG,
    t1_config: t1.PreprocessingConfig = t1.DEFAULT_CONFIG,
    t2_config: t2.SegmentationConfig = t2.DEFAULT_CONFIG,
) -> str:
    """Executes the end-to-end recognition pipeline."""
    target_size = getattr(model, "target_size", config.target_size)
    t2_config = replace(t2_config, target_size=target_size)

    mask = t1.preprocess_image(img_path, t1_config)
    _, crops = t2.segment_and_extract(mask, t2_config)

    if not crops:
        return ""

    raw_expression = _predict_characters(crops, model, config)
    return _format_result(raw_expression)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python task3_prediction.py <input_image> [model_type]")
        sys.exit(1)

    input_file = sys.argv[1]
    selected_model_type = sys.argv[2] if len(sys.argv) > 2 else "cnn"

    model_paths = {
        "cnn": "best_cnn_model.pth",
        "mobilenet_v3_small": "best_mobilenet_v3_small_model.pth",
        "resnet18": "best_resnet18_model.pth",
    }

    if selected_model_type not in model_paths:
        logger.error("Invalid model type. Supported types: cnn, mobilenet_v3_small, resnet18")
        sys.exit(1)

    pred_config = PredictionConfig(
        model_type=selected_model_type,
        model_path=model_paths[selected_model_type],
    )

    try:
        active_model = load_model(pred_config)
        output = predict_image(input_file, active_model, pred_config)
        print(output)
    except Exception as e:
        logger.error("Pipeline failure: %s", e)
        sys.exit(1)