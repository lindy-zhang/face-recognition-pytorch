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
    Docstring for unnormalize
    
    :param face_tensor: Description
    :type face_tensor: torch.Tensor
    :return: Description
    :rtype: Tensor
    """