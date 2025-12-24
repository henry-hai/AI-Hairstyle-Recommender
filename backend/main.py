import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import io
from PIL import Image

app = FastAPI()

# Enable CORS so React frontend can talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MediaPipe Face Mesh (The "Deep Learning" Model)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Barber Logic: Mapping Face Shapes to Recommendations
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

def analyze_face_shape(image):
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if not results.multi_face_landmarks:
        return "Unknown"

    landmarks = results.multi_face_landmarks[0].landmark
    h, w, _ = image.shape

    # Helper to get coordinates
    def get_coords(index):
        return np.array([landmarks[index].x * w, landmarks[index].y * h])

    # Key Landmarks (Indices based on MediaPipe mesh)
    # Cheekbones: 454 (Left), 234 (Right)
    # Jawline: 365 (Left), 136 (Right)
    # Face Height: 10 (Top), 152 (Bottom)
    # Forehead: 103 (Left), 333 (Right)

    cheek_width = np.linalg.norm(get_coords(454) - get_coords(234))
    jaw_width = np.linalg.norm(get_coords(365) - get_coords(136))
    face_height = np.linalg.norm(get_coords(10) - get_coords(152))
    forehead_width = np.linalg.norm(get_coords(103) - get_coords(333))

    # Ratios
    cheek_to_jaw = cheek_width / jaw_width
    height_to_cheek = face_height / cheek_width

    # Simple Classification Logic (tweak this based on barber experience)
    if height_to_cheek > 1.55:
        return "Oblong"
    elif cheek_width > forehead_width and cheek_width > jaw_width and cheek_to_jaw > 1.3:
        return "Diamond"
    elif abs(cheek_width - face_height) < 0.1 * face_height: # Roughly equal
        if jaw_width < cheek_width * 0.9:
            return "Round"
        else:
            return "Square"
    elif jaw_width < forehead_width:
        return "Heart"
    else:
        return "Oval"

@app.post("/analyze")
async def analyze_hairstyle(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 1. Computer Vision Step
    shape = analyze_face_shape(image)

    # 2. Recommendation Step
    recommendations = HAIRSTYLE_DATABASE.get(shape, {"styles": [], "description": "Could not determine."})

    return {
        "face_shape": shape,
        "description": recommendations["description"],
        "recommended_styles": recommendations["styles"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)