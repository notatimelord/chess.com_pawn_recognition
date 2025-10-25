import torch
from torchvision import transforms, models
from torch import nn
from PIL import Image
import numpy as np
import cv2
from pathlib import Path

# ==========================================================
# CONFIGURATION
# ==========================================================
MODEL_PATH = Path(r"\chess_cnn.pth")
BOARD_IMAGE = Path(r"\example.png")  
IMG_SIZE = 64
FILES = ['a','b','c','d','e','f','g','h']
RANKS = ['8','7','6','5','4','3','2','1']
# ==========================================================

# Load model
checkpoint = torch.load(MODEL_PATH, map_location="cpu")
class_names = checkpoint["class_names"]
print(f"✅ Loaded model with {len(class_names)} classes: {class_names}")

model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model.load_state_dict(checkpoint["model_state"])
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

board = Image.open(BOARD_IMAGE)
board_cv = cv2.imread(str(BOARD_IMAGE))
h, w = board.size[1], board.size[0]
cell_h, cell_w = h // 8, w // 8

# ==========================================================
# SPLIT INTO 64 SQUARES AND CLASSIFY
# ==========================================================
occupied = []

for r in range(8):
    for f in range(8):
        x1, y1 = f * cell_w, r * cell_h
        x2, y2 = (f + 1) * cell_w, (r + 1) * cell_h

        square_img = board.crop((x1, y1, x2, y2)).convert("RGB")
        input_tensor = transform(square_img).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            pred_idx = torch.argmax(output, dim=1).item()
            label = class_names[pred_idx]

        square_name = f"{FILES[f]}{RANKS[r]}"

        if label != "empty":
            occupied.append((square_name, label))
            cv2.rectangle(board_cv, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(board_cv, label, (x1+2, y1+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)

# ==========================================================
# SHOW RESULTS
# ==========================================================
print("\n===== ♟️ DETECTED CHESS PIECES =====")
for sq, lbl in occupied:
    print(f" → {lbl:12s} on {sq}")

print(f"\nTotal detected pieces: {len(occupied)}")

cv2.imshow("Detected board", board_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()

