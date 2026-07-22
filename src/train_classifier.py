"""
train_classifier.py

- loads the embeddings produced by embeddings.py
- splits them into train/test sets (80/20)
- trains an SVM to classify embeddings by identity + reports accuracy.

"""

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Environment + Path Configurations
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

EMBEDDINGS_PATH = PROJECT_ROOT / "output" / "embeddings.npz"
CLASSIFIER_PATH = PROJECT_ROOT / "output" / "classifier.pkl"

RANDOM_STATE = 42  # using fixed seed for reproducible train/test split and results

def load_embeddings(path: Path):
    data = np.load(path)
    # unpack .npz archive created in feature extraction stage
    return data["embeddings"], data["labels"], data["class_names"]

def main():
    embeddings, labels, class_names = load_embeddings(EMBEDDINGS_PATH)
    print(f"Loaded {embeddings.shape[0]} embeddings, {len(class_names)} identities")
    
    # stratify=labels: ensures each identity's images are split proportionally
    # between train/test, rather than risking (say) all of one person's photos
    # landing in the test set purely by chance. Important here because our
    # classes are imbalanced (53 to 530 images per person).
    
    # Split: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings,
        labels,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    print(f"Train: {X_train.shape[0]} images, Test: {X_test.shape[0]} images")

    # kernel='linear': embeddings are already well-separated by the embedding
    # model, so a linear decision boundary is typically enough
    classifier = SVC(kernel="linear", probability=True, random_state=RANDOM_STATE)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nTest accuracy: {accuracy:.4f}")

    print("\nPer-class report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Save both the trained classifier and the class name list together ->
    # need class_names at inference time to turn a predicted integer
    # label back into a person's name.
    CLASSIFIER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
        "classifier": classifier,
        "class_names": class_names,
        "train_embeddings": X_train,
        "train_labels": y_train,
        },
        CLASSIFIER_PATH,
    )
    print(f"\nSaved classifier to: {CLASSIFIER_PATH}")


if __name__ == "__main__":
    main()