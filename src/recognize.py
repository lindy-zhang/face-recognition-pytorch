"""
recognize.py

inference on a single new photo: detect + align the face,
embed it, and classify it against our known identities (or unknown)

python3 src/recognize.py path/to/photo.jpg
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

CLASSIFIER_PATH = PROJECT_ROOT / "output" / "classifier.pkl"

DETECT_DEVICE = "cpu"
EMBED_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

CONFIDENCE_THRESHOLD = 0.5

def load_pipeline():
    """
    Load all three trained components once
    """

    mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=DETECT_DEVICE)
    embedder = InceptionResnetV1(pretrained="vggface2").eval().to(EMBED_DEVICE)

    saved = joblib.load(CLASSIFIER_PATH)
    classifier = saved["classifier"]
    class_names = saved["class_names"]

    return mtcnn, embedder, classifier, class_names

def get_embedding(image_path: Path, mtcnn: MTCNN, embedder: InceptionResnetV1) -> np.ndarray | None:
    """
    Detect + align + embed a single image. Returns None if no face found
    """
    img = Image.open(image_path).convert("RGB")

    face_tensor = mtcnn(img)  # detection + alignment + resize + MTCNN's own internal normalization
    if face_tensor is None:
        return None

    face_tensor = face_tensor.unsqueeze(0).to(EMBED_DEVICE)  # add batch dim

    with torch.no_grad():
        embedding = embedder(face_tensor)

    return embedding.cpu().numpy()[0]

def recognize(image_path: Path) -> None:
    mtcnn, embedder, classifier, class_names = load_pipeline()

    embedding = get_embedding(image_path, mtcnn, embedder)

    if embedding is None:
        print("No face detected in image")
        return
    
    # predict_proba: probability distribution across all known classes.
    # reshape(1, -1): sklearn expects a 2D array (n_samples, n_features),
    # even for a single sample.
    probabilities = classifier.predict_proba(embedding.reshape(1, -1))[0]

    top_idx = np.argmax(probabilities)
    top_confidence = probabilities[top_idx]
    top_name = class_names[top_idx]

    print(f"\nTop prediction: {top_name} ({top_confidence:.2%} confidence)")

    if top_confidence < CONFIDENCE_THRESHOLD:
        print(f"Confidence below threshold ({CONFIDENCE_THRESHOLD:.0%}) -> Unknown")
    else:
        print(f"Result: {top_name}")

    # Show the full distribution too (in case it's useful)
    print("\nFull probability distribution:")
    for idx in np.argsort(probabilities)[::-1]:
        print(f"  {class_names[idx]:20s} {probabilities[idx]:.2%}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 src/recognize.py path/to/photo.jpg")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"File not found: {image_path}")
        sys.exit(1)

    recognize(image_path)