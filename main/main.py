from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import cv2
from mediapipe import solutions
import pickle

app = FastAPI()

# Habilitar CORS para cualquier solicitud
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar los modelos preentrenados
with open('proyecto1B.pkl', 'rb') as f:
    model = pickle.load(f)

with open('lean.pkl', 'rb') as f:
    lean_model = pickle.load(f)

with open('hips.pkl', 'rb') as f:
    hips_model = pickle.load(f)

mp_drawing = solutions.drawing_utils
mp_pose = solutions.pose
pose = mp_pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5)

current_stage = ''


def process_image(image):
    global current_stage

    # Convertir la imagen de BGR a RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Procesar la imagen para obtener los landmarks
    results = pose.process(image_rgb)

    # Dibujar los landmarks en la imagen
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

    # Inicializar resultados por defecto
    bodylang_prob = 0
    bodylang_class = ''
    hips_class = 'neutral'
    hips_prob = 0
    lean_class = 'neutral'
    lean_prob = 0
    advice_text = ''
    landmarks = []

    try:
        # Extraer coordenadas y visibilidad de cada landmark
        row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
        X = pd.DataFrame([row])

        # Predecir probabilidad y clase de la postura
        bodylang_prob = model.predict_proba(X)[0]
        bodylang_class = model.predict(X)[0]
        
        # Obtener la probabilidad más alta
        max_bodylang_prob = max(bodylang_prob)

        # Actualización de la lógica de estado
        if bodylang_class == "down" and bodylang_prob[bodylang_prob.argmax()] > 0.7:
            current_stage = "down"
        elif current_stage == "down" and bodylang_class == "up" and bodylang_prob[bodylang_prob.argmax()] > 0.7:
            current_stage = "up"

        # Predicción del análisis de forma
        # Para lean_prob
        lean_prob = np.array(lean_model.predict_proba(X)[0])
        lean_class = lean_model.predict(X)[0]
        max_lean_prob = max(lean_prob)

        # Para hips_prob
        hips_prob = np.array(hips_model.predict_proba(X)[0])
        hips_class = hips_model.predict(X)[0]
        max_hips_prob = max(hips_prob)

       # Guardar los landmarks para retornarlos en la respuesta
        landmarks = [{
            'x': res.x,
            'y': res.y,
            'z': res.z,
            'visibility': res.visibility
        } for res in results.pose_landmarks.landmark]

        # Determinar el mensaje de consejo
        if hips_class == "narrow" and lean_class == "right":
            advice_text = "Separa los pies e inclinate a la izquierda."
        elif hips_class == "narrow" and lean_class == "neutral":
            advice_text = "Separa los pies"
        elif hips_class == "narrow" and lean_class == "left":
            advice_text = "Separa los pies e inclinate a la derecha."
        elif hips_class == "neutral" and lean_class == "right":
            advice_text = "Inclinate a la izquierda"
        elif hips_class == "neutral" and lean_class == "neutral":
            advice_text = "Perfecto"
        elif hips_class == "neutral" and lean_class == "left":
            advice_text = "Inclinate a la derecha."
        elif hips_class == "wide" and lean_class == "right":
            advice_text = "Junta los pies e inclinate a la izquierda."
        elif hips_class == "wide" and lean_class == "neutral":
            advice_text = "Junta los pies"
        elif hips_class == "wide" and lean_class == "left":
            advice_text = "Junta los pies e inclinate a la derecha."

    except Exception as e:
        print("Error al procesar la imagen:", e)

    # Retornar los resultados en un diccionario
    return {
        "probability": max_bodylang_prob,
        "class": bodylang_class,
        "wide": hips_class,
        "wide_probability": max_hips_prob,
        "lean": lean_class,
        "lean_probability": max_lean_prob,
        "advice": advice_text,
        "landmarks": landmarks
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    results = process_image(image)
    return results


@app.post("/reset")
async def reset():
    global current_stage
    current_stage = ''
    return {"message": "Counter and state have been reset."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)