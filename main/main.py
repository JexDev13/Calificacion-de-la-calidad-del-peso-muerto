from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import pickle
import pandas as pd
from pydantic import BaseModel
from landmarks import landmarks
import mediapipe as mp

app = FastAPI()

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite solicitudes de cualquier origen
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos (POST, GET, etc.)
    allow_headers=["*"],  # Permite todos los encabezados
)

# Cargar modelos entrenados
with open('proyecto1B.pkl', 'rb') as f:
    model = pickle.load(f)

with open('lean.pkl', 'rb') as f:
    lean_model = pickle.load(f)

with open('hips.pkl', 'rb') as f:
    hips_model = pickle.load(f)

class PredictionRequest(BaseModel):
    image: bytes

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    results = process_image(image)
    return results

def process_image(image):
    pose = mp.solutions.pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5)
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
    X = pd.DataFrame([row], columns=landmarks)
    bodylang_prob = model.predict_proba(X)[0]
    bodylang_class = model.predict(X)[0]

    # Predicción del análisis de forma usando los modelos entrenados
    lean_prob = np.array(lean_model.predict_proba(X)[0])
    lean_class = lean_model.predict(X)[0]
    hips_prob = np.array(hips_model.predict_proba(X)[0])
    hips_class = hips_model.predict(X)[0]

    # Determinar el mensaje de consejo
    if hips_class == "narrow" and lean_class == "right":
        advice_text = "Necesitas ampliar tu postura para arreglarla e inclinarte hacia la izquierda para enderezar los hombros."
    elif hips_class == "narrow" and lean_class == "neutral":
        advice_text = "Necesitas ampliar tu postura para arreglarla y nada con tu inclinación, estás parejo."
    elif hips_class == "narrow" and lean_class == "left":
        advice_text = "Necesitas ampliar tu postura para arreglarla e inclinarte hacia la izquierda para enderezar los hombros."
    elif hips_class == "neutral" and lean_class == "right":
        advice_text = "No necesitas hacer nada para arreglar tu postura PERO inclinarse hacia la izquierda para enderezar los hombros."
    elif hips_class == "neutral" and lean_class == "neutral":
        advice_text = "No necesitas hacer nada para arreglar tu postura y nada con tu inclinación, estás parejo."
    elif hips_class == "neutral" and lean_class == "left":
        advice_text = "No necesitas hacer nada para arreglar tu postura PERO inclinarse hacia la derecha para enderezar los hombros."
    elif hips_class == "wide" and lean_class == "right":
        advice_text = "Necesitas juntar los pies para fijar la postura e inclinarte hacia la izquierda para enderezar los hombros."
    elif hips_class == "wide" and lean_class == "neutral":
        advice_text = "Necesitas juntar los pies para fijar la postura y nada con tu inclinación, estás parejo."
    elif hips_class == "wide" and lean_class == "left":
        advice_text = "Necesitas juntar los pies para fijar la postura e inclinarte hacia la derecha para enderezar los hombros."

    return {
        "class": bodylang_class,
        "probability": bodylang_prob[bodylang_prob.argmax()],
        "lean_class": lean_class,
        "lean_prob": lean_prob[lean_prob.argmax()],
        "hips_class": hips_class,
        "hips_prob": hips_prob[hips_prob.argmax()],
        "advice": advice_text,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)