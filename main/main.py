from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import pickle
import pandas as pd
from pydantic import BaseModel
from landmarks import landmarks
import mediapipe as mp
import asyncio

app = FastAPI()

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite solicitudes de cualquier origen
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos (POST, GET, etc.)
    allow_headers=["*"],  # Permite todos los encabezados
)

# Montar los archivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")

# Cargar modelos entrenados
with open('proyecto1B.pkl', 'rb') as f:
    model = pickle.load(f)

with open('lean.pkl', 'rb') as f:
    lean_model = pickle.load(f)

with open('hips.pkl', 'rb') as f:
    hips_model = pickle.load(f)

# Estado para manejar el modelo, cronómetro y frame
streaming = False
timer_running = False
last_frame = None

# Contador de repeticiones
rep_counter = {
    "current_class": None,
    "repetitions": 0
}

class PredictionRequest(BaseModel):
    image: bytes

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global last_frame
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Guarda el último frame capturado
    last_frame = image
    
    results = process_image(image)
    return results

@app.post("/start")
async def start():
    global streaming, timer_running
    if not streaming:
        streaming = True
        timer_running = True
        # Realiza una predicción inicial
        await predict_frame()
    return {"message": "Started"}

@app.post("/pause")
async def pause():
    global streaming
    streaming = False
    # Congela el modelo en el último frame capturado
    if last_frame is not None:
        process_image(last_frame)  # Procesa el último frame para congelar el estado
    return {"message": "Paused"}

@app.post("/reset")
async def reset():
    global streaming, timer_running, rep_counter
    streaming = False
    timer_running = False
    rep_counter["current_class"] = None
    rep_counter["repetitions"] = 0
    
    # Reinicia el modelo con una nueva predicción
    if last_frame is not None:
        await predict_frame()
    
    return {"message": "Reset"}

async def predict_frame():
    global last_frame
    if last_frame is not None:
        contents = cv2.imencode('.jpg', last_frame)[1].tobytes()
        # Simula un archivo para enviar en la solicitud
        class FakeUploadFile:
            def __init__(self, contents):
                self.file = contents
            async def read(self):
                return self.file
        fake_file = FakeUploadFile(contents)
        await predict(fake_file)

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

    # Actualizar contador de repeticiones
    global rep_counter
    if bodylang_class == "up":
        if rep_counter["current_class"] == "down":
            rep_counter["repetitions"] += 1
        rep_counter["current_class"] = "up"
    elif bodylang_class == "down":
        rep_counter["current_class"] = "down"

    return {
        "class": bodylang_class,
        "probability": bodylang_prob[bodylang_prob.argmax()],
        "lean_class": lean_class,
        "lean_prob": lean_prob[lean_prob.argmax()],
        "hips_class": hips_class,
        "hips_prob": hips_prob[hips_prob.argmax()],
        "advice": advice_text,
        "repetitions": rep_counter["repetitions"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
