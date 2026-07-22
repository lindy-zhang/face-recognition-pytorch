# Face Recognition Pipeline (PyTorch)

MTCNN (detection+alignment) → InceptionResnetV1/VGGFace2 (512-d embeddings) → SVM (classification), with distance-based rejection for unknown faces.

## Pipeline
1. `src/align_faces.py` — detect + align faces from `data/lfw_subset/`
2. `src/embeddings.py` — extract 512-d embeddings via InceptionResnetV1
3. `src/train_classifier.py` — train SVM on embeddings (97.95% test accuracy on 10-identity LFW subset)
4. `src/recognize.py` — end-to-end inference on a new photo

## Usage
python3 src/recognize.py path/to/photo.jpg

## Known limitations
- SVM confidence alone can't detect out-of-gallery faces; added nearest-neighbor
  distance check as a second signal
- Trained/tested only on well-lit, frontal LFW photos — not validated on
  low-quality or non-frontal images