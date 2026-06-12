# AI Hairstyle Recommender

A multimodal AI app that pairs **computer vision** with a **GPT-4o** reasoning layer to recommend hairstyles. A MediaPipe pipeline reads front-facing facial geometry and classifies the face shape; a rule engine grounded in **14 years of professional barbering** selects the flattering cuts; and GPT-4o turns that result into a personalized, plain-English explanation. Each recommendation is paired with a reference photo.

> **Division of labor:** the face detection, geometric ratios, and cut recommendations are 100% deterministic code (the barbering expertise). The LLM only *explains* those recommendations in natural language — it never decides the face shape or invents cuts.

## Project Structure
* **/backend** — FastAPI server: MediaPipe Face Mesh for landmark detection + face-shape classification, plus an optional GPT-4o explanation layer.
* **/frontend** — React UI: drag-and-drop upload, recommendation display, and a reference-photo grid.

## Technical Implementation
* **Computer Vision** — MediaPipe Face Mesh extracts 468 facial landmarks and computes Cheek-to-Jaw and Height-to-Cheek ratios.
* **Recommendation Engine** — Rule-based classifier maps geometry to one of six face shapes and a curated cut list, grounded in professional barbering standards.
* **LLM Layer (optional)** — GPT-4o, constrained to the engine's output, generates a personalized rationale. Degrades gracefully to a static description when no API key is set.
* **Frontend** — Responsive React UI with a custom CSS design system.

## Setup and Installation

### Prerequisites
* Python 3.8+
* Node.js & npm

### Backend Setup
1. `cd backend`
2. Copy env template: `cp .env.example .env` (optionally add your `OPENAI_API_KEY` to enable GPT-4o)
3. Install dependencies: `pip install -r requirements.txt`
4. Start the server: `python main.py`

### Frontend Setup
1. `cd frontend`
2. (optional) `cp .env.example .env` to point at a non-default backend URL
3. Install dependencies: `npm install`
4. Start the app: `npm start`

## Personalizing the Reference Photos
Placeholder images live in `frontend/public/haircuts/`. To use your own barbering gallery, replace the `.jpg` files there (keep the same filenames) — no code changes needed. The style-to-image mapping lives in `frontend/src/data/haircuts.js`.
