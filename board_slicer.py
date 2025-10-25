import cv2
from pathlib import Path

# ----------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------
SOURCE_DIR = Path(r"C:\board")         # folder with chess.png, chess4.png
OUTPUT_DIR = Path(r"C:\crops") # will be created automatically

FILES = ['a','b','c','d','e','f','g','h']
RANKS = ['8','7','6','5','4','3','2','1']

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------
# CROP 8x8 GRID FROM ALL IMAGES
# ----------------------------------------------------------
found_images = list(SOURCE_DIR.glob("*.png")) + list(SOURCE_DIR.glob("*.jpg"))
if not found_images:
    print(f"[⚠️] No .png or .jpg files found in {SOURCE_DIR}")
else:
    print(f"Found {len(found_images)} image(s) to crop...")

for img_path in found_images:
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[!] Could not open {img_path}")
        continue

    h, w = img.shape[:2]
    cell_h, cell_w = h // 8, w // 8

    print(f"Cropping {img_path.name} → 64 squares...")

    for r in range(8):
        for f in range(8):
            y1, y2 = r * cell_h, (r + 1) * cell_h
            x1, x2 = f * cell_w, (f + 1) * cell_w
            crop = img[y1:y2, x1:x2]
            sq_name = f"{FILES[f]}{RANKS[r]}"
            out_name = f"{img_path.stem}_{sq_name}.png"
            cv2.imwrite(str(OUTPUT_DIR / out_name), crop)

print(f"✅ Cropping complete. Saved images to {OUTPUT_DIR}")
