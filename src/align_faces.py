# PREPROCESS IMAGES
# 1. Walk every person's folder in data/lfw_subset/
# 2. For each image: detect + align + crop w/ MTCNN
# 3. Save aligned crop to output/aligned/<PersonName>/<same_filename>
# 4. Track and log any images where no face was found
# 5. Print a summary at the end (X processed, Y failed)

"""
align_faces.py

Walks a directory of face images organized as:
    input_dir/PersonName/image.jpg

Detects, aligns, and crops the largest face in each image using MTCNN,
then writes the result (a normalized 160x160 tensor -> saved as an image)
to a mirrored directory structure under output_dir.

"""

import os
from pathlib import Path

import torch
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision.utils import save_image

# Prevent path errors (that I ran into earlier)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

INPUT_DIR = PROJECT_ROOT / "data" / "lfw_subset"
OUTPUT_DIR = PROJECT_ROOT / "output" / "aligned"

# Use CPU rather than MPS
DEVICE = "cpu"

def build_detector() -> MTCNN:
    """
    keep_all=False: only keep the single most-confident face per image.
        LFW images are already mostly single-subject portraits, so this
        matches the article's "find the largest/most prominent face" logic.
    
    post_process=True (default): normalizes pixel values for the embedding
        model's expected input distribution.
    """
    return MTCNN(image_size=160, margin=20, keep_all=False, device=DEVICE)

def unnormalize(face_tensor: torch.Tensor) -> torch.Tensor:
    """
    Bc MTCNN's output tensor is normalized to roughly [-1, 1] for the embedding
    model -> to save it as a viewable image file,  need to map it back to
    [0, 1], which is what image codecs (save_image / PIL) expect.
    """
    return (face_tensor + 1) / 2

def process_dataset(input_dir: Path, output_dir: Path) -> None:
    mtcnn = build_detector()

    output_dir.mkdir(parents=True, exist_ok=True)

    total_images = 0
    succeeded = 0
    failed = [] # list of (path, reason) tuples

    person_dirs = sorted(p for p in input_dir.iterdir() if p.is_dir())
    for person_dir in person_dirs:
        person_name = person_dir.name
        out_person_dir = output_dir / person_name
        out_person_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(person_dir.glob("*.jpg"))
        
        for img_path in image_paths:
            total_images += 1
            try:
                # Enforce 3-channel RGB
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                failed.append((img_path, f"could not open image: {e}"))
                continue

            face_tensor = mtcnn(img)

            if face_tensor is None:
                failed.append((img_path, "no face detected"))
            
            out_path = out_person_dir / img_path.name
            save_image(unnormalize(face_tensor), str(out_path))
            succeeded += 1
        
        print(f"[{person_name}] {len(image_paths)}")
    
    # --- Summary ---------------------------------------------------
    print("\n=== Alignment Summary ===")
    print(f"Total images:  {total_images}")
    print(f"Succeeded:     {succeeded}")
    print(f"Failed:        {len(failed)}")

    if failed:
        print("\nFailures:")
        for path, reason in failed[:20]:  # cap console spam
            print(f"  {path}: {reason}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")

if __name__ == "__main__":
    process_dataset(INPUT_DIR, OUTPUT_DIR)