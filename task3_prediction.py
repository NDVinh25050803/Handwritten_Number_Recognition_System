"""
Task 3 — Image Prediction and Evaluation (HNRS)

Integrates Task 1 (preprocessing) and Task 2 (segmentation) to classify
individual character segments using trained PyTorch models[cite: 3].
Implements the required extensions for recognising alphanumeric characters
and arithmetic symbols, and conditionally evaluates valid mathematical
expressions[cite: 3].

Pipeline
--------
1. Model Initialization: Loads a specified PyTorch model architecture and
   its corresponding state dictionary.
2. Image Processing: Executes Task 1 preprocessing and Task 2 segmentation
   to extract individual character segments.
3. Tensor Conversion: Converts extracted numpy arrays into batched PyTorch
   tensors, applying necessary normalization and channel adjustments.
4. Inference: Computes class probabilities and maps the highest scoring
   indices to their corresponding character representations.
5. Expression Evaluation: Attempts to parse and evaluate the concatenated
   prediction string as a mathematical expression[cite: 3]. Retains the raw
   string if it contains alphabetical characters or terminates in an invalid
   syntax.
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
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


# Configuration

@dataclass
class PredictionConfig:
    """Tunable parameters for the prediction pipeline.

    Attributes:
        model_type: Architecture selection.
        model_path: Path to the PyTorch state dictionary.
        device: Hardware accelerator selection.
        target_size: Expected input spatial dimensions for the model.
        in_channels: Expected number of input channels (1 for grayscale,
            3 for RGB).
        class_map: Mapping from output tensor indices to string characters.
            Must cover 0-9, a-z, A-Z, and symbols (+, -, *, /, (, )) for
            the full scope of the project extensions[cite: 3].
    """

    model_type: Literal["cnn", "mobilenet_v3_small", "resnet18"] = "cnn"
    model_path: str = "best_cnn_model.pth"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    target_size: int = 28
    in_channels: int = 1

    class_map: list[str] = field(default_factory=lambda: [
        *list("0123456789"),
        *list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"),
        "+", "-", "*", "/", "(", ")"
    ])


DEFAULT_CONFIG = PredictionConfig()


# Model Definitions

class SimpleCNN(nn.Module):
    """Baseline CNN architecture for character recognition."""
    def __init__(self, num_classes: int, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def load_model(config: PredictionConfig) -> nn.Module:
    """Instantiate the model architecture and load trained weights.

    Args:
        config: Prediction configuration dictating model architecture and path.

    Returns:
        PyTorch nn.Module set to evaluation mode.
    """
    num_classes = len(config.class_map)

    if config.model_type == "cnn":
        model = SimpleCNN(num_classes=num_classes, in_channels=config.in_channels)
    elif config.model_type == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        if config.in_channels != 3:
            model.features[0][0] = nn.Conv2d(config.in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif config.model_type == "resnet18":
        model = models.resnet18(weights=None)
        if config.in_channels != 3:
            model.conv1 = nn.Conv2d(config.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model_type: {config.model_type}")

    state_dict = torch.load(config.model_path, map_location=torch.device(config.device))
    model.load_state_dict(state_dict)
    model.to(config.device)
    model.eval()

    return model


# Evaluation Logic

def _format_result(expression: str) -> str:
    """Attempt arithmetic evaluation of a parsed string[cite: 3].

    Args:
        expression: The concatenated predicted characters.

    Returns:
        String detailing the evaluation result or the original string if unevaluable.
    """
    if re.search(r'[a-zA-Z]', expression):
        return expression

    try:
        parsed = ast.parse(expression, mode='eval')
        result = eval(compile(parsed, filename="<string>", mode="eval"))
        
        if isinstance(result, float) and result.is_integer():
            result = int(result)
            
        return f"{expression} = {result}"
    except Exception:
        return expression


# Public API

def predict_image(
    img_path: str,
    model: nn.Module,
    config: PredictionConfig = DEFAULT_CONFIG,
    t1_config: t1.PreprocessingConfig = t1.DEFAULT_CONFIG,
    t2_config: t2.SegmentationConfig = t2.DEFAULT_CONFIG,
) -> str:
    """Execute the full pipeline on a target image file.

    Args:
        img_path: Location of the input image.
        model: Pre-loaded PyTorch model in evaluation mode.
        config: Tunable prediction parameters.
        t1_config: Preprocessing parameters.
        t2_config: Segmentation parameters.

    Returns:
        The final interpreted string or calculated equation result.
    """
    t2_config.target_size = config.target_size

    mask = t1.preprocess_image(img_path, t1_config)
    _, crops = t2.segment_and_extract(mask, t2_config)

    if not crops:
        return ""

    batch_np = np.stack(crops).astype(np.float32) / 255.0
    
    if config.in_channels == 3:
        batch_np = np.stack([batch_np] * 3, axis=1)
    else:
        batch_np = np.expand_dims(batch_np, axis=1)

    batch_tensor = torch.from_numpy(batch_np).to(config.device)

    with torch.no_grad():
        outputs = model(batch_tensor)
        _, predicted = torch.max(outputs, 1)

    indices = predicted.cpu().numpy()
    
    char_list = []
    for idx in indices:
        if idx < len(config.class_map):
            char_list.append(config.class_map[idx])
        else:
            char_list.append("?")
            
    raw_expression = "".join(char_list)
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
        "resnet18": "best_resnet18_model.pth"
    }

    if selected_model_type not in model_paths:
        logger.error("Invalid model type. Choose from: cnn, mobilenet_v3_small, resnet18")
        sys.exit(1)

    pred_config = PredictionConfig(
        model_type=selected_model_type,
        model_path=model_paths[selected_model_type]
    )

    try:
        active_model = load_model(pred_config)
        output = predict_image(input_file, active_model, pred_config)
        print(output)
    except Exception as e:
        logger.error("Pipeline failure: %s", e)
        sys.exit(1)
