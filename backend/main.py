import os
import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from features import extract_features

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration (env-driven so the same code runs locally and in production)
# ---------------------------------------------------------------------------
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

app = FastAPI(title="AI Hairstyle Recommender")

# Enable CORS so the React frontend can talk to the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MediaPipe Face Mesh (the computer-vision model)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Load the trained face-shape classifier once at startup. If the artifact is
# missing or fails to load, the app keeps working on the threshold rules below,
# so a missing model can never take the endpoint down.
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def load_shape_model():
    try:
        import joblib

        path = os.path.join(MODELS_DIR, "face_shape_sklearn.joblib")
        model = joblib.load(path)
        print(f"[model] loaded classifier from {path}")
        return model
    except Exception as exc:  # noqa: BLE001 - degrade to rules, never crash on load
        print(f"[model] classifier unavailable, using rule-based fallback: {exc}")
        return None


shape_model = load_shape_model()

# ---------------------------------------------------------------------------
# Barber Logic: face shapes -> recommendations (the source of truth).
# This is grounded in 14 years of professional barbering experience. The LLM
# never overrides this - it only explains these specific recommendations.
# ---------------------------------------------------------------------------
HAIRSTYLE_DATABASE = {
    "Oval": {
        "description": "Balanced proportions. You can wear almost anything.",
        "styles": ["Pompadour", "Quiff", "Textured Crop Top/Mullet", "Buzz Cut", "Combover w/ Side Part"]
    },
    "Square": {
        "description": "Strong jawline. Keep sides short to accentuate jaw, volume on top to lengthen.",
        "styles": ["Undercut", "High Fade with Length on Top", "Slick Back", "Faux Hawk"]
    },
    "Round": {
        "description": "Soft angles. Needs angles and height to elongate the face.",
        "styles": ["High Volume Quiff", "Flat Top", "Combover with Volume", "Textured Crop Top/Mullet with extra Volume", "2 Block Haircut - Short on Sides"]
    },
    "Diamond": {
        "description": "Wide cheekbones, narrow forehead/jaw. Needs width at forehead.",
        "styles": ["Pompadour/Quiff", "Textured Crop Top/Mullet", "Buzz Cut", "Combover w/ Side Part"]
    },
    "Oblong": {
        "description": "Face is longer than it is wide. Avoid too much height.",
        "styles": ["Buzz Cut", "Short Combover", "Crew Cut", "Textued Fringe/Crop Top", "2 Block Haircut"]
    },
    "Heart": {
        "description": "Wide forehead, narrow chin. Add volume to chin or fringe to forehead.",
        "styles": ["Textured Crop Top", "Textured Quiff", "Side Part", "Textured Mullet", "Beard if Possible"]
    }
}


def classify_by_rules(landmarks, w, h):
    """Threshold-based fallback classifier from a handful of raw width ratios.

    Kept as the safety net for when the trained model is not available, so the
    endpoint always returns a shape.
    """
    def get_coords(index):
        return np.array([landmarks[index].x * w, landmarks[index].y * h])

    # Key landmarks (MediaPipe mesh indices)
    # Cheekbones: 454 / 234 | Jawline: 365 / 136 | Height: 10 / 152 | Forehead: 103 / 333
    cheek_width = np.linalg.norm(get_coords(454) - get_coords(234))
    jaw_width = np.linalg.norm(get_coords(365) - get_coords(136))
    face_height = np.linalg.norm(get_coords(10) - get_coords(152))
    forehead_width = np.linalg.norm(get_coords(103) - get_coords(333))

    cheek_to_jaw = cheek_width / jaw_width
    height_to_cheek = face_height / cheek_width

    if height_to_cheek > 1.55:
        return "Oblong"
    if cheek_width > forehead_width and cheek_width > jaw_width and cheek_to_jaw > 1.3:
        return "Diamond"
    if abs(cheek_width - face_height) < 0.1 * face_height:  # roughly equal
        return "Round" if jaw_width < cheek_width * 0.9 else "Square"
    if jaw_width < forehead_width:
        return "Heart"
    return "Oval"


def analyze_face_shape(image):
    """Extract face landmarks and classify the face shape.

    Returns (shape, ratios, confidence). Ratios are the geometric measurements
    used for classification and are surfaced to the user and fed to the LLM as
    grounding context. Confidence is the model's probability for the predicted
    shape, or None when the rule-based fallback produced the result.
    """
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        return "Unknown", {}, None

    landmarks = results.multi_face_landmarks[0].landmark
    h, w, _ = image.shape

    vector, ratios = extract_features(landmarks, image.shape)

    if shape_model is not None:
        try:
            proba = shape_model.predict_proba([vector])[0]
            idx = int(np.argmax(proba))
            shape = str(shape_model.classes_[idx])
            confidence = round(float(proba[idx]), 2)
            return shape, ratios, confidence
        except Exception as exc:  # noqa: BLE001 - fall back to rules, never crash
            print(f"[model] prediction failed, using rule-based fallback: {exc}")

    return classify_by_rules(landmarks, w, h), ratios, None


def generate_explanation(shape, recommendation, ratios, confidence=None):
    """Optional: turn the classified result into a personalized, plain-English
    explanation.

    The model is constrained to the recommendations produced above. It only
    rephrases them and never introduces new advice. If no API key is configured
    or the call fails, this returns None and the frontend falls back to the
    static description.
    """
    if not OPENAI_API_KEY or shape == "Unknown":
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY)
        styles = ", ".join(recommendation["styles"])
        confidence_line = (
            f"Classifier confidence in this shape: {confidence}.\n"
            if confidence is not None
            else ""
        )
        prompt = (
            f"A computer-vision system classified this client's face shape as '{shape}'.\n"
            f"Geometric ratios measured: {ratios}.\n"
            f"{confidence_line}"
            f"The barber's rule for this shape: \"{recommendation['description']}\"\n"
            f"The barber recommends these cuts: {styles}.\n\n"
            "Write 2-3 warm, confident sentences, as a master barber speaking to the client, "
            "explaining WHY these specific cuts flatter their face shape. "
            "Only reference the cuts listed above - do not invent new styles or contradict the "
            "barber's recommendation. Be concrete about how the cuts play off their geometry."
        )

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a master barber with 14 years of experience. "
                                              "You explain recommendations clearly and never invent advice "
                                              "beyond what you are given."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=180,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:  # noqa: BLE001 - never break the core product over the LLM
        print(f"[LLM] explanation skipped: {exc}")
        return None


@app.get("/")
def health():
    return {
        "status": "ok",
        "llm_enabled": bool(OPENAI_API_KEY),
        "model_enabled": shape_model is not None,
    }


@app.post("/analyze")
async def analyze_hairstyle(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Detect the face and classify its shape.
    shape, ratios, confidence = analyze_face_shape(image)

    # Map the shape to the recommended cuts.
    recommendation = HAIRSTYLE_DATABASE.get(
        shape, {"styles": [], "description": "Could not determine a face shape. Try a clear, front-facing photo."}
    )

    # Optional plain-English explanation of the recommendation.
    explanation = generate_explanation(shape, recommendation, ratios, confidence)

    return {
        "face_shape": shape,
        "description": recommendation["description"],
        "explanation": explanation,      # personalized GPT-4o text, or null
        "ratios": ratios,
        "confidence": confidence,        # model probability, or null on fallback
        "recommended_styles": recommendation["styles"],
    }


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)