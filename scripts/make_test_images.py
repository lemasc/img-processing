"""Create flat, noise, and resized test images. Usage: make_test_images.py [photo]"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
OUT = Path(__file__).resolve().parent.parent / "images"
SIZES = {1: (1280, 800), 6: (3000, 2000), 24: (6000, 4000), 48: (8000, 6000)}

def main():
    rng = np.random.default_rng(0)
    for mp, (w, h) in SIZES.items():
        Image.fromarray(np.full((h, w, 3), 128, np.uint8)).save(OUT / f"flat_{mp}mp.png")
        Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)).save(OUT / f"noise_{mp}mp.png")
    if len(sys.argv) > 1:
        src = Image.open(sys.argv[1]).convert("RGB")
        for mp, size in SIZES.items():
            src.resize(size, Image.LANCZOS).save(OUT / f"photo_{mp}mp.png")

if __name__ == "__main__":
    main()
