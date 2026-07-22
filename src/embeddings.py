"""
embeddings.py

Loads aligned face crops (from align_faces.py's output) and runs them
through a pretrained InceptionResnetV1 (trained on VGGFace2) to produce
a 512-dimensional embedding vector per face.

2 photos of the same person should map to nearby vectors, and photos
of different people should map to distant vectors even for identities
the model never saw during training
-> Works bc InceptionResnetV1 was trained w/ a metric-learning objective (triplet loss in the original
FaceNet paper) that explicitly optimizes for that separation property,
rather than for classifying a fixed set of people.

Output: output/embeddings.npz containing
    - embeddings: (N, 512) float32 array
    - labels: (N,) int array (index into class_names)
    - class_names: list[str], person names in index order
    - paths: (N,) list of source image paths, for traceability/debugging
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

ALIGNED_DIR = PROJECT_ROOT / "output" / "aligned"
EMBEDDINGS_PATH = PROJECT_ROOT / "output" / "embeddings.npz"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

BATCH_SIZE = 32 # Process 32 images at a time

def build_dataset(aligned_dir: Path) -> datasets.ImageFolder:
    """
    ImageFolder expects: root/ClassName/image.jpg - which is exactly the
    structure align_faces.py produced. It automatically infers class labels
    from subfolder names and builds an index -> class_name mapping for us
    (accessible as dataset.classes), so we don't need to parse folder names
    ourselves.

    transform pipeline:
      1. ToTensor(): PIL Image [0,255] uint8 -> torch tensor [0,1] float32,
         shape (C, H, W). This undoes the JPEG-saved [0,1] range from disk.
      2. fixed_image_standardization: maps [0,1] -> the exact input
         distribution InceptionResnetV1 was trained on.
    """

    transform = transforms.Compose([
    transforms.ToTensor(),                       # PIL -> [0,1] tensor
    transforms.Lambda(lambda x: x * 255),         # undo the [0,1] scaling -> back to [0,255]
    fixed_image_standardization,                  # now correctly maps [0,255] -> model's expected range
])
    return datasets.ImageFolder(root=str(aligned_dir), transform=transform)

def extract_embeddings(dataset: datasets.ImageFolder) -> tuple[np.ndarray, np.ndarray, list[str]]:
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE) # Use pre-trained weights

    all_embeddings = []
    all_labels = []

    # torch.no_grad() -> disable gradient tracking since only doing inference, no need for backprop
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(DEVICE)
            batch_embeddings = model(images) # shape: (batch_size, 512)

            all_embeddings.append(batch_embeddings.cpu().numpy())
            all_labels.append(labels.numpy())

            print(f"Processed batch {batch_idx + 1}/{len(loader)}")

    embeddings = np.concatenate(all_embeddings, axis=0) # shape: (N, 512)
    labels = np.concatenate(all_labels, axis=0) # shape: (N, )

    return embeddings, labels, dataset.classes

def main():
    dataset = build_dataset(ALIGNED_DIR)
    print(f"Found {len(dataset)} images across {len(dataset.classes)} identities")
    print(f"Using device: {DEVICE}")

    embeddings, labels, class_names = extract_embeddings(dataset)

    # image paths, in the same order the dataset iterated them, for
    # traceability if we ever need to debug a specific embedding
    paths = [str(path) for path, _ in dataset.samples]

    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        EMBEDDINGS_PATH,
        embeddings=embeddings,
        labels=labels,
        class_names=np.array(class_names),
        paths=np.array(paths),
    )

    print(f"\nSaved {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
    print(f"Written to: {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()