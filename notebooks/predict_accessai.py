import os
import sys
import csv
from pathlib import Path
from copy import deepcopy

import pandas as pd
import torch
from torch import nn
from PIL import Image, ImageOps
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent

if not torch.cuda.is_available():
    os.environ["XFORMERS_DISABLED"] = "1"

sys.path.insert(0, str(ROOT / "dinov2"))
from dinov2.models.vision_transformer import vit_base

IMAGE_DIR = Path.home() / "Downloads" / "Sidewalk Photos"
PRETRAINED = ROOT / "dinov2_vitb14_reg4_pretrain.pth"
CHECKPOINT = ROOT / "notebooks/models/validated-dino-cls-b-surfaceproblem-tags-best.pth"
OUTPUT = ROOT / "results/accessai_model_predictions.csv"

TAGS = {
    0: "brick_cobblestone",
    1: "bumpy",
    2: "construction",
    3: "cracks",
    4: "grass",
    5: "height_difference",
    6: "narrow_sidewalk",
    7: "rail_tram_track",
    8: "sand_gravel",
    9: "uncovered_manhole",
    10: "uneven_slanted",
    11: "utility_panel",
    12: "very_broken",
}

class ResizeAndPad:
    def __call__(self, image):
        image = transforms.Resize((256, 256))(image)
        w = (14 - image.width % 14) % 14
        h = (14 - image.height % 14) % 14
        return transforms.Pad(
            (w // 2, h // 2, w - w // 2, h - h // 2)
        )(image)
    
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = vit_base(
            patch_size=14, img_size=526, init_values=1.0,
            num_register_tokens=4, block_chunks=0
        )
        backbone.load_state_dict(torch.load(PRETRAINED, map_location="cpu"))
        self.transformer = deepcopy(backbone)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 13)
        )
    def forward(self, x):
        x = self.transformer(x)
        return self.classifier(self.transformer.norm(x))
    
device = torch.device("cpu")

model = Classifier()
weights = torch.load(CHECKPOINT, map_location="cpu")
if "model_state_dict" in weights:
    weights = weights["model_state_dict"]
model.load_state_dict(weights)
model.to(device).eval()

preprocess = transforms.Compose([
    ResizeAndPad(),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

MANUAL_LABELS = ROOT / "data/manual_labels.csv"

manual = pd.read_csv(MANUAL_LABELS, dtype=str)

photo_manifest = manual[["photo_id", "photo_filename"]].copy()

for column in ["photo_id", "photo_filename"]:
    photo_manifest[column] = photo_manifest[column].str.strip()

if photo_manifest.isna().any().any():
    raise ValueError("Missing photo IDs or filenames in manual_labels.csv")

if photo_manifest["photo_id"].duplicated().any():
    raise ValueError("Duplicate photo IDs in manual_labels.csv")

if photo_manifest["photo_filename"].duplicated().any():
    raise ValueError("Duplicate photo filenames in manual_labels.csv")

image_files = []

for _, row in photo_manifest.iterrows():
    photo_id = row["photo_id"]
    filename = row["photo_filename"]

    if Path(filename).name != filename:
        raise ValueError(f"Invalid photo filename: {filename}")

    image_path = IMAGE_DIR / filename

    if not image_path.is_file():
        raise FileNotFoundError(f"Missing image: {image_path}")

    image_files.append((photo_id, image_path))

print(f"Found {len(image_files)} labeled photos.")

results = []

for photo_id, path in image_files:
    with Image.open(path) as pil_image:
        pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")
        input_tensor = preprocess(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor).squeeze(0).cpu()

    scores = torch.sigmoid(logits).tolist()
    logits = logits.tolist()

    row = {"photo_id": photo_id, "photo_filename": path.name}

    for index, tag in TAGS.items():
        row[f"score_{tag}"] = scores[index]
        row[f"logit_{tag}"] = logits[index]

    results.append(row)
    print("Processed:", path.name)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(results).to_csv(OUTPUT, index=False)
print("Saved predictions to:", OUTPUT)
    