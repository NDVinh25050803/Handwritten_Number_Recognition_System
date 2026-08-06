"""
Task 4 / GUI — Streamlit Front-End (HNRS)

Wraps the three pipeline stages into one interactive app:
  Task 1 (task1_preprocessing) -> binarized mask
  Task 2 (task2_segmentation)  -> per-character crops
  Task 3 (task3_prediction)    -> classification + expression evaluation

Two image acquisition modes:
  - upload a single image file directly
  - type a number and auto-compose it from a folder of single-digit
    reference images bundled in the repo (see build_number_from_digits)

Every pipeline hyper-parameter is exposed as a sidebar control, so the
user can tune preprocessing/segmentation/model choice and see the effect
on each intermediate stage live.
"""

import os
import re
import tempfile

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

import task1_preprocessing as t1
import task2_segmentation as t2
import task3_prediction as t3

st.set_page_config(page_title="Handwritten Number Recognition", layout="wide")

MODEL_PATHS = {
    "cnn": "best_cnn_model.pth",
    "mobilenet_v3_small": "best_mobilenet_v3_small_model.pth",
    "resnet18": "best_resnet18_model.pth",
}

# No GPU on the deployment target, so device is a fixed constant rather
# than a selectable UI option.
DEVICE = "cpu"

# All three architectures were trained on 28x28 source images, so Task 2
# always segments to 28x28 regardless of which model is selected. Task 3
# (predict_with_topk) upsamples 28 -> each model's native input size to
# match what it saw during training.
DIGIT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "digit")


# Image acquisition helpers

def save_upload_to_disk(uploaded_file) -> str:
    """Write a Streamlit UploadedFile to a temp path, since t1/t2/t3
    take file paths rather than in-memory bytes."""
    suffix = os.path.splitext(uploaded_file.name)[1] or ".png"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return tmp.name


def extract_digit_key(filename: str) -> str:
    """Extract the digit a reference image represents from its filename,
    e.g. '7.png' -> '7', 'digit_0.png' -> '0'.
    """
    stem = filename.rsplit(".", 1)[0]
    return re.sub(r"\D", "", stem)


@st.cache_data
def load_digit_folder(folder: str) -> dict:
    """Map each digit to its reference image path by scanning the digit
    folder bundled in the repo."""
    digit_to_path = {}
    if not os.path.isdir(folder):
        return digit_to_path
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        key = extract_digit_key(fname)
        if key:
            digit_to_path[key] = os.path.join(folder, fname)
    return digit_to_path


def build_number_from_digits(digit_to_path: dict, number_str: str, gap_px: int = 0) -> str:
    """Compose a number image by placing each digit's reference image
    left to right, in the order the digits appear in number_str, reusing
    an image whenever its digit repeats.

    Images keep their original color (grayscale conversion is Task 1's
    job) and are aligned to a common height. No gap is inserted between
    tiles by default — Task 1's illumination normalization flattens any
    brightness mismatch between adjacent tiles before thresholding.

    Returns the path to the composed image on disk.

    Raises:
        ValueError: if number_str contains a digit with no matching
            reference image.
    """
    missing = sorted(set(d for d in number_str if d not in digit_to_path))
    if missing:
        raise ValueError(f"No reference image for digit(s): {', '.join(missing)}")

    imgs = [np.array(Image.open(digit_to_path[d]).convert("RGB")) for d in number_str]
    target_h = max(im.shape[0] for im in imgs)

    resized = []
    for im in imgs:
        scale = target_h / im.shape[0]
        new_w = max(1, int(im.shape[1] * scale))
        resized.append(cv2.resize(im, (new_w, target_h)))

    pieces = resized
    if gap_px > 0:
        gap = np.full((target_h, gap_px, 3), 255, dtype=np.uint8)  # white paper background
        pieces = []
        for i, im in enumerate(resized):
            pieces.append(im)
            if i != len(resized) - 1:
                pieces.append(gap)
    canvas = cv2.cvtColor(np.concatenate(pieces, axis=1), cv2.COLOR_RGB2BGR)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    cv2.imwrite(tmp.name, canvas)
    return tmp.name


# Cached model loading — weights only reloaded when architecture or path changes

@st.cache_resource
def get_model(model_type: str, model_path: str):
    config = t3.PredictionConfig(model_type=model_type, model_path=model_path, device=DEVICE)
    return t3.load_model(config)


# Inference with confidence scores

def predict_with_topk(crops, model, config: t3.PredictionConfig, top_k: int = 3):
    """Classify each character crop and return its top_k (class, probability)
    predictions, sorted highest first.

    Preprocessing mirrors task3_prediction._predict_characters exactly —
    same resize target, channel handling, and per-model mean/std — the
    only difference is returning the full softmax distribution per
    character instead of collapsing straight to the argmax string.
    """
    in_channels = getattr(model, "in_channels", config.in_channels)
    class_map = getattr(model, "class_map", config.class_map)
    target_size = getattr(model, "target_size", config.target_size)

    processed = []
    for crop in crops:
        if crop.shape != (target_size, target_size):
            crop = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        processed.append(crop)

    batch = np.stack(processed).astype(np.float32) / 255.0
    if in_channels == 3:
        # Grayscale -> 3 identical channels, then ImageNet stats (backbones
        # are ImageNet-pretrained).
        batch = np.repeat(batch[:, np.newaxis, :, :], 3, axis=1)
        mean_vals, std_vals = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        # Custom single-channel CNN, normalized with its own dataset stats.
        batch = batch[:, np.newaxis, :, :]
        mean_vals, std_vals = [0.1532], [0.3114]

    batch_tensor = torch.from_numpy(batch).to(config.device)
    mean_tensor = torch.tensor(mean_vals, device=config.device).view(1, -1, 1, 1)
    std_tensor = torch.tensor(std_vals, device=config.device).view(1, -1, 1, 1)
    batch_tensor = (batch_tensor - mean_tensor) / std_tensor

    with torch.no_grad():
        probs = torch.softmax(model(batch_tensor), dim=1).cpu().numpy()

    per_char_topk = []
    for row in probs:
        top_indices = np.argsort(row)[::-1][:top_k]
        per_char_topk.append([
            (class_map[i] if i < len(class_map) else "?", float(row[i]))
            for i in top_indices
        ])
    return per_char_topk


# Sidebar — parameters

st.sidebar.header("1. Image source")
source_mode = st.sidebar.radio("Acquisition mode", ["Upload one image", "Compose from digit folder"])

st.sidebar.header("2. Model")
model_type = st.sidebar.selectbox("Architecture", list(MODEL_PATHS.keys()))
model_path = st.sidebar.text_input("Weights file", MODEL_PATHS[model_type])

st.sidebar.header("3. Preprocessing (Task 1)")
illum_frac = st.sidebar.slider("Illumination close fraction", 0.0, 0.2, 0.05, 0.01)
blur_ksize = st.sidebar.slider("Blur kernel size", 1, 9, 3, 2)
border_frac = st.sidebar.slider("Border sample fraction", 0.01, 0.2, 0.05, 0.01)
min_noise = st.sidebar.slider("Min noise area (Task 1)", 0, 200, 20, 5)

st.sidebar.header("4. Segmentation (Task 2)")
merge_stacked = st.sidebar.checkbox("Merge stacked components (e.g. ÷, :)", True)
x_overlap = st.sidebar.slider("Min x-overlap ratio", 0.0, 1.0, 0.3, 0.05)
max_gap = st.sidebar.slider("Max stack gap ratio", 0.0, 2.0, 0.8, 0.1)
min_comp_area = st.sidebar.slider("Min component area (Task 2)", 0, 200, 40, 5)
pad_ratio = st.sidebar.slider("Padding ratio", 0.0, 0.6, 0.2, 0.05)

t1_config = t1.PreprocessingConfig(
    illum_close_frac=illum_frac, blur_ksize=blur_ksize,
    border_frac=border_frac, min_noise_area=min_noise,
)
t2_config = t2.SegmentationConfig(
    merge_stacked=merge_stacked, x_overlap_min_ratio=x_overlap,
    max_gap_ratio=max_gap, min_component_area=min_comp_area,
    pad_ratio=pad_ratio,
    target_size=28,
)

# Main — image acquisition

st.title("Handwritten Number Recognition System")

img_path = None
if source_mode == "Upload one image":
    uploaded = st.file_uploader("Upload a handwritten number/expression image", type=["png", "jpg", "jpeg"])
    if uploaded:
        img_path = save_upload_to_disk(uploaded)
else:
    digit_to_path = load_digit_folder(DIGIT_FOLDER)
    if not digit_to_path:
        st.error(f"No digit reference images found in '{DIGIT_FOLDER}'. "
                 f"Make sure the 'digit' folder is committed to the repo next to app.py.")
    else:
        st.caption(f"Reference digits available: {', '.join(sorted(digit_to_path))}")
        number_str = st.text_input("Number to compose (digits only) — press Enter to build", "")
        number_str = re.sub(r"\D", "", number_str)  # strip anything that isn't a digit

        if number_str:
            try:
                img_path = build_number_from_digits(digit_to_path, number_str)
            except ValueError as e:
                st.error(str(e))

if img_path is None:
    st.info("Provide an image to run the pipeline.")
    st.stop()

st.image(img_path, caption="Acquired image", use_container_width=True)

# Run pipeline

run = st.button("Run recognition", type="primary")
if not run:
    st.stop()

# Task 1 — preprocessing, with every intermediate stage for visualisation
mask, stages = t1.preprocess_image(img_path, t1_config, debug=True)

st.subheader("Task 1 — Preprocessing stages")
cols = st.columns(len(stages))
for col, (name, stage_img) in zip(cols, stages.items()):
    col.image(stage_img, caption=name, use_container_width=True)

# Task 2 — segmentation
boxes, crops = t2.segment_and_extract(mask, t2_config)

st.subheader("Task 2 — Segmentation")
if not crops:
    st.warning("No characters detected — try relaxing the noise/area thresholds.")
    st.stop()

vis = t2.draw_segmentation(mask, boxes)
st.image(vis, caption=f"{len(boxes)} segment(s) detected", channels="BGR", width=500)

# Task 3 — recognition with per-character confidence
try:
    model = get_model(model_type, model_path)
except Exception as e:
    st.error(f"Could not load model weights '{model_path}': {e}")
    st.stop()

pred_config = t3.PredictionConfig(model_type=model_type, model_path=model_path, device=DEVICE)
topk_predictions = predict_with_topk(crops, model, pred_config, top_k=3)

# Each crop shown with its top-3 predicted classes, one prediction per line
st.subheader("Task 2 — Segments  /  Task 3 — Per-character confidence")
char_cols = st.columns(len(crops))
for col, crop, topk in zip(char_cols, crops, topk_predictions):
    col.image(crop, use_container_width=True)
    for cls, prob in topk:
        col.markdown(f"**{cls}** — {prob:.1%}")

# Final string from each character's top-1 prediction, evaluated as
# arithmetic if it contains no letters
raw_prediction = "".join(char_topk[0][0] for char_topk in topk_predictions)
result = t3._format_result(raw_prediction)

st.subheader("Final prediction")
st.markdown(f"## `{result}`")
