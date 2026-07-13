"""Geometric feature extraction from MediaPipe face-mesh landmarks.

This is the single source of truth for turning a set of face landmarks into the
ratio vector the classifier uses. Both the offline training pipeline and the live
/analyze endpoint call extract_features, so the features are identical at train
time and serve time.

Every feature is a ratio of Euclidean distances (or an angle), which makes it
invariant to image scale and to in-plane head rotation. The features do not
correct for out-of-plane head pose, which is the known ceiling of single-photo
2D geometry.
"""

import numpy as np

# Landmark clusters (MediaPipe face-mesh indices). Each width is measured between
# the averaged position of a small cluster on each side rather than a single
# point, which steadies the measurement against per-landmark jitter.
_FOREHEAD_LEFT = (103, 67, 109)
_FOREHEAD_RIGHT = (332, 297, 338)
_CHEEK_LEFT = (234, 93)
_CHEEK_RIGHT = (454, 323)
_JAW_LEFT = (136, 172)
_JAW_RIGHT = (365, 397)
_CHIN_LEFT = (149, 148)
_CHIN_RIGHT = (378, 377)
_TOP = 10          # top of the forehead
_CHIN_TIP = 152    # bottom of the chin
_JAW_CORNER_LEFT = 172
_JAW_CORNER_RIGHT = 397
_CHEEK_POINT_LEFT = 234
_CHEEK_POINT_RIGHT = 454

# Ordered feature names. This order defines the vector layout and must not change
# without retraining, since the serialized model expects this exact ordering.
FEATURE_NAMES = [
    "height_to_cheek",
    "cheek_to_jaw",
    "forehead_to_cheek",
    "jaw_to_forehead",
    "height_to_jaw",
    "height_to_forehead",
    "chin_to_jaw",
    "cheek_to_fj_mean",
    "jaw_angle",
]


def _point(landmarks, indices, w, h):
    """Average pixel position of one or more landmark indices."""
    if isinstance(indices, int):
        indices = (indices,)
    pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in indices])
    return pts.mean(axis=0)


def _angle(vertex, a, b):
    """Angle in degrees at `vertex` between the rays to points `a` and `b`."""
    v1 = a - vertex
    v2 = b - vertex
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0
    cos = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def extract_features(landmarks, image_shape):
    """Turn face-mesh landmarks into the ordered ratio vector and a display dict.

    Args:
        landmarks: sequence of MediaPipe landmarks (each with .x, .y in [0, 1]).
        image_shape: the image's (height, width, ...) so normalized coordinates
            can be scaled back to pixels with the correct aspect ratio.

    Returns:
        (vector, feature_dict) where vector is a float32 np.ndarray in
        FEATURE_NAMES order, and feature_dict maps each name to a rounded value
        for display and for grounding the explanation layer.
    """
    h, w = image_shape[0], image_shape[1]

    forehead_l = _point(landmarks, _FOREHEAD_LEFT, w, h)
    forehead_r = _point(landmarks, _FOREHEAD_RIGHT, w, h)
    cheek_l = _point(landmarks, _CHEEK_LEFT, w, h)
    cheek_r = _point(landmarks, _CHEEK_RIGHT, w, h)
    jaw_l = _point(landmarks, _JAW_LEFT, w, h)
    jaw_r = _point(landmarks, _JAW_RIGHT, w, h)
    chin_l = _point(landmarks, _CHIN_LEFT, w, h)
    chin_r = _point(landmarks, _CHIN_RIGHT, w, h)
    top = _point(landmarks, _TOP, w, h)
    chin_tip = _point(landmarks, _CHIN_TIP, w, h)

    forehead_width = np.linalg.norm(forehead_r - forehead_l)
    cheek_width = np.linalg.norm(cheek_r - cheek_l)
    jaw_width = np.linalg.norm(jaw_r - jaw_l)
    chin_width = np.linalg.norm(chin_r - chin_l)
    face_height = np.linalg.norm(top - chin_tip)

    # Small epsilon guards against a zero denominator on a degenerate detection.
    eps = 1e-6
    fj_mean = (forehead_width + jaw_width) / 2.0

    jaw_corner_l = _point(landmarks, _JAW_CORNER_LEFT, w, h)
    jaw_corner_r = _point(landmarks, _JAW_CORNER_RIGHT, w, h)
    cheek_pt_l = _point(landmarks, _CHEEK_POINT_LEFT, w, h)
    cheek_pt_r = _point(landmarks, _CHEEK_POINT_RIGHT, w, h)
    jaw_angle = 0.5 * (
        _angle(jaw_corner_l, cheek_pt_l, chin_tip)
        + _angle(jaw_corner_r, cheek_pt_r, chin_tip)
    )

    values = {
        "height_to_cheek": face_height / (cheek_width + eps),
        "cheek_to_jaw": cheek_width / (jaw_width + eps),
        "forehead_to_cheek": forehead_width / (cheek_width + eps),
        "jaw_to_forehead": jaw_width / (forehead_width + eps),
        "height_to_jaw": face_height / (jaw_width + eps),
        "height_to_forehead": face_height / (forehead_width + eps),
        "chin_to_jaw": chin_width / (jaw_width + eps),
        "cheek_to_fj_mean": cheek_width / (fj_mean + eps),
        "jaw_angle": jaw_angle,
    }

    vector = np.array([values[name] for name in FEATURE_NAMES], dtype=np.float32)
    feature_dict = {name: round(float(values[name]), 2) for name in FEATURE_NAMES}
    return vector, feature_dict