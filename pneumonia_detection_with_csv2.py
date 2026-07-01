import os
import sys
import time
import json
import shutil
import torch
import clip
import pandas as pd
import re

from PIL import Image
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# =========================
# CONFIG
# =========================

img_dir = "input_directory"  # input directory which has xray images
output_json = "pneumonia_results.json"
output_csv = "prediction_resultsdate_time2.csv"

output_dir = "output_xray"
os.makedirs(output_dir, exist_ok=True)

# =========================
# LOAD MODEL
# =========================

weights = hf_hub_download(
    repo_id="jayanthapoojary1989/rsna-pneumonia-yolov8s",
    filename="pytorch_yolov8_model.pt"
)

detector = YOLO(weights)
detector.names[0] = "Pneumonia"

print("Class names:", detector.names)

# =========================
# CLIP X-RAY VALIDATION
# =========================

device = "cpu"

clip_model, preprocess = clip.load(
    "ViT-B/32",
    device=device
)

xray_texts = clip.tokenize([
    "a medical x-ray image",
    "a radiology x-ray scan",
    "a medical radiograph",
    "a chest x-ray"
]).to(device)

non_xray_texts = clip.tokenize([
    "a normal photograph",
    "a document or paper",
    "a whiteboard",
    "a screenshot",
    "CT image",
    "MRI image"
]).to(device)


def extract_datetime(filename):

    try:

        # Remove extension
        name = os.path.splitext(filename)[0]

        # Example:
        # 500502352_IMG_20250729_131953

        parts = name.split("_")

        if len(parts) >= 4:

            date_str = parts[-2]
            time_str = parts[-1]

            # Validate lengths
            if len(date_str) == 8 and len(time_str) == 6:

                date_taken = (
                    f"{date_str[:4]}-"
                    f"{date_str[4:6]}-"
                    f"{date_str[6:]}"
                )

                time_taken = (
                    f"{time_str[:2]}:"
                    f"{time_str[2:4]}:"
                    f"{time_str[4:]}"
                )

                return date_taken, time_taken

    except Exception as e:
        print(f"Date extraction error for {filename}: {e}")

    return "", ""

def is_xray(image_path):

    image = preprocess(
        Image.open(image_path).convert("RGB")
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        img_f = clip_model.encode_image(image)
        xray_f = clip_model.encode_text(xray_texts)
        non_f = clip_model.encode_text(non_xray_texts)

    return (
        (img_f @ xray_f.T).mean()
        >
        (img_f @ non_f.T).mean()
    )

# =========================
# PROCESS
# =========================

start_time = time.time()

results_json = []
results_table = []

if not os.path.isdir(img_dir):
    print("Input directory does not exist")
    sys.exit(0)

for fname in sorted(os.listdir(img_dir)):

    if not fname.lower().endswith(
        (".png", ".jpg", ".jpeg")
    ):
        continue

    path = os.path.join(img_dir, fname)

    date_taken, time_taken = extract_datetime(fname)
    print(
    f"{fname} -> Date: {date_taken}, Time: {time_taken}"
    )
    output_image_path = os.path.join(
        output_dir,
        fname
    )

    # ---------------------
    # Non X-Ray
    # ---------------------

    if not is_xray(path):

        shutil.copy2(path, output_image_path)

        results_json.append({
            "filename": fname,
            "result": "Non-XRay",
            "confidence": 0,
            "region": None,
            "image": output_image_path
        })

        results_table.append({
            "Image Name": fname,
            "Date": date_taken,
            "Time": time_taken,
            "Prediction Score (%)": 0,
            "Prediction": "Non-XRay",
            "x1": "",
            "y1": "",
            "x2": "",
            "y2": ""
        })

        continue

    # ---------------------
    # Detect Pneumonia
    # ---------------------

    results = detector(path, verbose=False)
    result = results[0]

    # ---------------------
    # Normal
    # ---------------------

    if len(result.boxes) == 0:

        shutil.copy2(path, output_image_path)

        results_json.append({
            "filename": fname,
            "result": "Normal",
            "confidence": 0,
            "region": None,
            "image": output_image_path
        })

        results_table.append({
            "Image Name": fname,
            "Date": date_taken,
            "Time": time_taken,
            "Prediction Score (%)": 0,
            "Prediction": "Normal",
            "x1": "",
            "y1": "",
            "x2": "",
            "y2": ""
        })

        continue

    # ---------------------
    # Pneumonia Found
    # ---------------------

    best_idx = result.boxes.conf.argmax()

    confidence = float(
        result.boxes.conf[best_idx]
    ) * 100

    box = (
        result.boxes.xyxy[best_idx]
        .cpu()
        .numpy()
        .tolist()
    )

    x1, y1, x2, y2 = map(int, box)

    result.save(
        filename=output_image_path
    )

    results_json.append({
        "filename": fname,
        "result": f"{confidence:.1f}% Pneumonia Detected",
        "confidence": round(confidence, 1),
        "region": {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        },
        "image": output_image_path
    })

    results_table.append({
        "Image Name": fname,
        "Date": date_taken,
        "Time": time_taken,
        "Prediction Score (%)": round(confidence, 2),
        "Prediction": "Pneumonia",
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2
    })

# =========================
# SAVE JSON
# =========================

elapsed = time.time() - start_time

final_output = {
    "processing_time_seconds": round(elapsed, 2),
    "total_images": len(results_json),
    "results": results_json
}

with open(output_json, "w") as f:
    json.dump(final_output, f, indent=4)

# =========================
# SAVE CSV
# =========================

df = pd.DataFrame(results_table)
df.to_csv(output_csv, index=False)

print(f"[INFO] Done in {elapsed:.2f} sec")
print(f"[INFO] JSON saved: {output_json}")
print(f"[INFO] CSV saved: {output_csv}")
print(f"[INFO] Images saved in: {output_dir}")
print("\\nPrediction Summary:")
print(df)
