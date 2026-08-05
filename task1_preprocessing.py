"""
Task 1 — Image Preprocessing (HNRS)

Converts a photographed or scanned image of a handwritten number or
expression into a clean binary mask — white ink on a black background,
at the image's original resolution. The output feeds directly into
Task 2 (task2_segmentation.segment_characters); resizing to a fixed
model input size happens there, after each character is cropped, not
here.

Scope: handwriting on a plain, unlined background. Ruled or grid paper
should be cropped or masked out before calling this module — detecting
and stripping printed lines is out of scope.

Pipeline
--------
1. Grayscale conversion.
2. Illumination normalization — flattens shadows, vignetting, and
   uneven lighting so a single global threshold works across the whole
   photo (see _normalize_illumination).
3. A light Gaussian blur, to suppress pixel-level sensor/JPEG noise
   before thresholding.
4. Otsu binarization, with ink/background polarity decided by sampling
   the image border (see _binarize).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Iterator, Literal, overload

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# Configuration

@dataclass
class PreprocessingConfig:
    """Tunable parameters for the preprocessing pipeline.

    Attributes:
        illum_close_frac: morphological closing kernel size, as a
            fraction of the shorter image side, used to estimate the
            paper background for illumination normalization. Must be
            wider than the thickest ink feature in the photo so the
            closing can fully recover the background under it; set to 0
            to disable normalization on input already known to be
            evenly lit.
        blur_ksize: odd kernel size for the pre-threshold denoising
            blur. Values <= 1 disable it.
        border_frac: width of the border band sampled to decide ink
            polarity, as a fraction of the shorter image side.
        min_noise_area: connected components smaller than this many
            pixels are dropped as noise. Kept deliberately below the
            size of any plausible character fragment — Task 2 performs
            the finer, merge-aware noise judgment afterwards.
        assume_dark_ink: "auto" decides polarity per image from the
            border sample. Pass True to force dark ink on light paper,
            or False to force light ink on a dark background.
    """

    illum_close_frac: float = 0.05
    blur_ksize: int = 3
    border_frac: float = 0.05
    min_noise_area: int = 20
    assume_dark_ink: str | bool = "auto"


DEFAULT_CONFIG = PreprocessingConfig()


# Pipeline steps

def _normalize_illumination(gray: np.ndarray, close_frac: float) -> np.ndarray:
    """Flatten uneven background lighting before thresholding.

    Args:
        gray: grayscale input image.
        close_frac: closing kernel size as a fraction of min(height, width).

    Returns:
        uint8 grayscale image with background brightness flattened.
    """
    if close_frac <= 0:
        return gray

    h, w = gray.shape
    k = max(int(min(h, w) * close_frac), 3)
    if k % 2 == 0:
        k += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    
    # Prevents zero division and eliminates redundant float32 allocations
    # by applying scaling and type casting directly within OpenCV.
    background[background == 0] = 1
    return cv2.divide(gray, background, scale=255.0, dtype=cv2.CV_8U)


def _smooth(gray: np.ndarray, ksize: int) -> np.ndarray:
    """Light Gaussian blur applied just before thresholding.

    Args:
        gray: grayscale (or illumination-normalized) input image.
        ksize: odd blur kernel size. Values <= 1 disable this step.

    Returns:
        uint8 blurred image, same shape as input.
    """
    if ksize <= 1:
        return gray
    k = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(gray, (k, k), 0)


def _border_is_background(raw_binary: np.ndarray, border_frac: float) -> bool:
    """Decide which class of an un-oriented binary split is background.

    Args:
        raw_binary: uint8 0/255 image from an un-oriented Otsu split.
        border_frac: border band width, as a fraction of the shorter
            image side.

    Returns:
        True if the bright class is the majority in the border band
        (bright = background, dark = ink); False otherwise.
    """
    h, w = raw_binary.shape
    b = max(1, int(min(h, w) * border_frac))
    border_pixels = np.concatenate([
        raw_binary[:b, :].ravel(),
        raw_binary[-b:, :].ravel(),
        raw_binary[:, :b].ravel(),
        raw_binary[:, -b:].ravel(),
    ])
    bright = int(np.count_nonzero(border_pixels))
    return bright >= border_pixels.size - bright


def _binarize(gray: np.ndarray, assume_dark_ink: str | bool, border_frac: float) -> np.ndarray:
    """Otsu binarization, oriented so ink is white (255) on black (0).

    Args:
        gray: preprocessed grayscale image.
        assume_dark_ink: "auto", True, or False.
        border_frac: forwarded to _border_is_background when "auto".

    Returns:
        uint8 0/255 image, white = ink, black = background.
    """
    if assume_dark_ink == "auto":
        _, raw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if _border_is_background(raw, border_frac):
            return cv2.bitwise_not(raw)
        return raw

    flag = cv2.THRESH_BINARY_INV if assume_dark_ink else cv2.THRESH_BINARY
    _, binary = cv2.threshold(gray, 0, 255, flag + cv2.THRESH_OTSU)
    return binary


def _remove_coarse_noise(mask: np.ndarray, min_noise_area: int) -> np.ndarray:
    """Drop connected components smaller than min_noise_area pixels.

    Args:
        mask: uint8 0/255 binary mask.
        min_noise_area: minimum pixel count for a component to survive.

    Returns:
        uint8 0/255 mask with small components removed.
    """
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]  # skip label 0 (background)
    valid_labels = np.where(areas >= min_noise_area)[0] + 1
    clean = np.zeros_like(mask)
    clean[np.isin(labels, valid_labels)] = 255
    return clean


# Public API

@overload
def preprocess_image(
    img_path: str,
    config: PreprocessingConfig = DEFAULT_CONFIG,
    *,
    debug: Literal[False] = False,
) -> np.ndarray: ...

@overload
def preprocess_image(
    img_path: str,
    config: PreprocessingConfig = DEFAULT_CONFIG,
    *,
    debug: Literal[True],
) -> tuple[np.ndarray, dict[str, np.ndarray]]: ...

def preprocess_image(
    img_path: str,
    config: PreprocessingConfig = DEFAULT_CONFIG,
    *,
    debug: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    """Convert a handwriting photo/scan into a clean binary mask.

    Args:
        img_path: path to the input image.
        config: tunable pipeline parameters. See PreprocessingConfig.
        debug: if True, also return every intermediate stage.

    Returns:
        mask: uint8 array (H, W), 0/255, white = ink. Feed this
            directly into task2_segmentation.segment_characters.
        stages: dict of intermediate masks by stage name — only if
            debug=True.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file exists but cannot be decoded as an image.
    """
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"No such file: {img_path}")

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not decode image (unsupported format or corrupt): {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flattened = _normalize_illumination(gray, config.illum_close_frac)
    smoothed = _smooth(flattened, config.blur_ksize)
    binary = _binarize(smoothed, config.assume_dark_ink, config.border_frac)
    mask = _remove_coarse_noise(binary, config.min_noise_area)

    if debug:
        stages = {
            "grayscale": gray,
            "illumination_normalized": flattened,
            "blurred": smoothed,
            "otsu_binary": binary,
            "final": mask,
        }
        return mask, stages

    return mask


def preprocess_folder(
    folder_path: str,
    config: PreprocessingConfig = DEFAULT_CONFIG,
) -> Iterator[tuple[str, np.ndarray]]:
    """Yield (filename, mask) for every image file in a folder.

    Args:
        folder_path: directory containing .png/.jpg/.jpeg images.
        config: forwarded to preprocess_image().

    Yields:
        (filename, mask) tuples, each mask at its original resolution.
    """
    for fname in sorted(os.listdir(folder_path)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        path = os.path.join(folder_path, fname)
        try:
            mask = preprocess_image(path, config)
        except (FileNotFoundError, ValueError, cv2.error) as e:
            logger.warning("Skipped %s: %s", fname, e)
            continue
        yield fname, mask


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python task1_preprocessing.py <input_image> <output_image>")
        sys.exit(1)

    input_path, output_path = sys.argv[1], sys.argv[2]
    result_mask = preprocess_image(input_path)
    cv2.imwrite(output_path, result_mask)
    logger.info("Saved preprocessed image to: %s", output_path)
