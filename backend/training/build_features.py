"""Build the training features table from a labeled face-shape image dataset.

Walks a directory of class-named subfolders, runs the same MediaPipe face-mesh
configuration the app uses, extracts the geometric ratio features, and writes a
CSV of feature columns plus a label column. Images where no face is detected are
skipped and counted, so we can see whether any class came out thin.

Usage:
    python build_features.py --data-dir /path/to/dataset --out ../data/features.csv
"""

import argparse
import csv
import os
import sys
from collections import Counter

import cv2
import mediapipe as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from features import FEATURE_NAMES, extract_features  # noqa: E402

# Folder name (lowercased) -> canonical face-shape label. Folders not in this map
# (for example "pear" and "triangle") are ignored, since the app supports these
# six shapes only.
LABEL_MAP = {
    "oval": "Oval",
    "round": "Round",
    "square": "Square",
    "heart": "Heart",
    "oblong": "Oblong",
    "rectangular": "Oblong",
    "rectangle": "Oblong",
    "diamond": "Diamond",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def label_for(path):
    """Return the canonical label for an image based on its parent folder name."""
    parent = os.path.basename(os.path.dirname(path)).lower()
    return LABEL_MAP.get(parent)


def iter_images(data_dir):
    for root, _dirs, files in os.walk(data_dir):
        for name in files:
            if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                yield os.path.join(root, name)


def main():
    parser = argparse.ArgumentParser(description="Extract face-shape features to CSV.")
    parser.add_argument("--data-dir", required=True, help="Root of the labeled dataset.")
    parser.add_argument("--out", required=True, help="Output CSV path.")
    args = parser.parse_args()

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True
    )

    kept = Counter()
    skipped_no_face = Counter()
    skipped_unreadable = Counter()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(FEATURE_NAMES + ["label"])

        for path in iter_images(args.data_dir):
            label = label_for(path)
            if label is None:
                continue

            image = cv2.imread(path)
            if image is None:
                skipped_unreadable[label] += 1
                continue

            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                skipped_no_face[label] += 1
                continue

            landmarks = results.multi_face_landmarks[0].landmark
            vector, _ = extract_features(landmarks, image.shape)
            writer.writerow([f"{v:.6f}" for v in vector] + [label])
            kept[label] += 1

    face_mesh.close()

    labels = sorted(set(kept) | set(skipped_no_face) | set(skipped_unreadable))
    print(f"\nWrote {sum(kept.values())} rows to {args.out}\n")
    print(f"{'label':<10}{'kept':>8}{'no_face':>10}{'unreadable':>12}")
    for label in labels:
        print(f"{label:<10}{kept[label]:>8}{skipped_no_face[label]:>10}{skipped_unreadable[label]:>12}")
    print(
        f"\nTotals: kept={sum(kept.values())} "
        f"no_face={sum(skipped_no_face.values())} "
        f"unreadable={sum(skipped_unreadable.values())}"
    )


if __name__ == "__main__":
    main()