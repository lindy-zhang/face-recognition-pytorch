"""
diagnose_embeddings.py — sanity check: do embeddings actually separate
identities, or did something break upstream?

Idea: for a well-behaved embedding space, distances between two photos of
the SAME person should typically be smaller than distances between photos
of DIFFERENT people. If that's not true on average, the embeddings
themselves carry little identity signal, and no classifier can fix that.
"""

import numpy as np
from pathlib import Path

data = np.load(Path(__file__).resolve().parent.parent / "output" / "embeddings.npz")
embeddings = data["embeddings"]
labels = data["labels"]

# Normalize embeddings to unit length so we're measuring angle/cosine-like
# distance, not raw magnitude -- standard practice for face embeddings.
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
normed = embeddings / norms

# Sample pairs to keep this fast
rng = np.random.default_rng(42)
n = len(embeddings)
n_pairs = 2000

same_dists = []
diff_dists = []

for _ in range(n_pairs):
    i, j = rng.integers(0, n, size=2)
    if i == j:
        continue
    dist = np.linalg.norm(normed[i] - normed[j])
    if labels[i] == labels[j]:
        same_dists.append(dist)
    else:
        diff_dists.append(dist)

print(f"Same-identity pairs sampled:      {len(same_dists)}")
print(f"Different-identity pairs sampled: {len(diff_dists)}")
print(f"Mean distance, SAME identity:     {np.mean(same_dists):.4f}")
print(f"Mean distance, DIFFERENT identity:{np.mean(diff_dists):.4f}")
print(f"\nEmbedding value stats: min={embeddings.min():.3f}, max={embeddings.max():.3f}, mean={embeddings.mean():.3f}, std={embeddings.std():.3f}")