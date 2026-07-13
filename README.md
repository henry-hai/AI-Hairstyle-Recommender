# AI Hairstyle Recommender

A multimodal app that pairs **computer vision** with a **GPT-4o** reasoning layer to recommend men's hairstyles. A MediaPipe pipeline reads front-facing facial geometry, a **trained machine-learning classifier** predicts the face shape from geometric ratios, a rule engine grounded in **14 years of professional barbering** selects the flattering cuts, and GPT-4o turns that result into a personalized, plain-English explanation. Each recommendation is paired with a reference photo.

> **Division of labor:** face detection, the geometric features, the face-shape classifier, and the cut recommendations are all deterministic code. The LLM only *explains* the result in natural language. It never decides the face shape or invents cuts.

## Project Structure
* **/backend** - FastAPI server: MediaPipe Face Mesh for landmark detection, a geometric feature extractor, a trained face-shape classifier with a rule-based fallback, and an optional GPT-4o explanation layer.
* **/frontend** - React UI: drag-and-drop upload, recommendation display with a confidence readout, and a reference-photo grid.

## Technical Implementation
* **Computer Vision** - MediaPipe Face Mesh extracts 468 facial landmarks. A single shared function (`extract_features`) turns them into nine geometric ratios, used identically at training time and at request time.
* **Face-shape classifier** - A scikit-learn RandomForest predicts one of six shapes with a confidence value. A small PyTorch network is trained alongside it as a benchmark. If the model artifact is missing or fails, the endpoint falls back to the original threshold rules, so it never hard-fails.
* **Recommendation Engine** - Maps the predicted shape to a curated cut list grounded in professional barbering standards.
* **LLM Layer (optional)** - GPT-4o, constrained to the engine's output, generates a personalized rationale. Degrades gracefully to a static description when no API key is set.
* **Frontend** - Responsive React UI with a custom CSS design system.

## The face-shape classifier

The earlier version classified the face with a short chain of hand-tuned `if`/`else` thresholds on two ratios. That logic is kept as a safety net, but the primary classifier is now trained on real, human-labeled data.

### Geometric features
Every feature is a ratio of Euclidean distances (or an angle) between averaged clusters of landmarks. Ratios of distances are invariant to image scale and to in-plane head rotation, and averaging a small cluster per side steadies each measurement against per-landmark jitter. The nine features are:

`height_to_cheek`, `cheek_to_jaw`, `forehead_to_cheek`, `jaw_to_forehead`, `height_to_jaw`, `height_to_forehead`, `chin_to_jaw`, `cheek_to_fj_mean`, and `jaw_angle`.

Together they capture overall elongation, the three competing widths (forehead, cheekbone, jaw), chin taper, and jaw angularity, which are the traits that separate the six shapes.

### Data
Training uses a public Kaggle face-shape image set covering the six supported shapes. Five classes hold roughly 1,000 images each; Diamond is small at 75 images, so it is the weakest class. The raw images are not committed; only the extracted numeric features (`backend/data/features.csv`) are.

The labels skew female. Because classification runs on sex-neutral geometric ratios rather than raw pixels, the shape relationships carry across sexes far better than an image-based model would, with a mild distributional skew worth noting (male jaws average heavier and squarer).

### Models and results
Both models train on the identical stratified train/test split (seed 42) and the identical feature set, and both are scored on the same held-out test set.

| Model | Role | Test accuracy |
|---|---|---|
| RandomForest (scikit-learn) | primary | 0.526 |
| MLP (PyTorch) | benchmark | 0.471 |

The six-class baseline is 0.167. On the training set, five-fold cross-validation gives 0.555 for the RandomForest and 0.464 for LogisticRegression, which is why the forest is the primary. Classical ML is the right tool here: the feature space is a handful of structured ratios, and the neural net does not beat it, which is the expected result. The PyTorch model is a deliberate benchmark, not a serving path. Full per-class precision, recall, F1, and confusion matrices are in `backend/models/metrics.md`.

Confidence comes from the forest's own `predict_proba`. Post-hoc probability calibration was tried and rejected because it drove the tiny Diamond class to zero predictions, which is a worse product than the forest's native, reasonably smooth probabilities.

### Honest limits
* All ratios come from a single 2D photo, so out-of-plane head pose (turning or tilting toward the camera) is an inherent ceiling. Front-facing photos give the best results.
* Diamond is the weakest class because of its small sample size and heavy geometric overlap with Heart and Oblong. Its per-class scores are the lowest, and `metrics.md` states this plainly.
* Face-shape labels are inherently subjective, which caps how much signal any model can extract from geometry alone.

## Setup and Installation

### Prerequisites
* Python 3.8+
* Node.js & npm

### Backend Setup
1. `cd backend`
2. Copy env template: `cp .env.example .env` (optionally add your `OPENAI_API_KEY` to enable GPT-4o)
3. Install dependencies: `pip install -r requirements.txt`
4. Start the server: `python main.py`

The trained model artifact ships in `backend/models/`, so the classifier works out of the box. With no artifact present, the server automatically falls back to the threshold rules.

### Frontend Setup
1. `cd frontend`
2. (optional) `cp .env.example .env` to point at a non-default backend URL
3. Install dependencies: `npm install`
4. Start the app: `npm start`

## Retraining from scratch

1. Set up a Kaggle API token (`~/.kaggle/access_token` or `~/.kaggle/kaggle.json`) and download the face-shape dataset.
2. Install the training dependencies: `pip install -r requirements-train.txt`
3. Extract features: `python training/build_features.py --data-dir <dataset> --out data/features.csv`
4. Train and evaluate both models: `python training/train_models.py --features data/features.csv --out-dir models`

Step 4 writes the serialized model, the PyTorch weights and scaler, the class order, and the metrics report into `backend/models/`.
