import tkinter as tk 
import customtkinter as ck

import pandas as pd
import numpy as np 
import pickle

import mediapipe as mp
import cv2
from PIL import Image, ImageTk

from landmarks import landmarks

window = tk.Tk()
window.geometry("480x850") 
window.title("GYM")
ck.set_appearance_mode("dark")

# Etiquetas y cajas para probabilidad, clase y contador de repeticiones
classLabel = ck.CTkLabel(window, height=30, width=120, font=("Arial", 14), text_color="black", padx=10) #height=40, width=120
classLabel.place(x=180, y=5)
classLabel.configure(text='Class')

counterLabel = ck.CTkLabel(window, height=30, width=120, font=("Arial", 14), text_color="black", padx=10)
counterLabel.place(x=320, y=5)
counterLabel.configure(text='Rep Counter')

probLabel = ck.CTkLabel(window, height=30, width=120, font=("Arial", 14), text_color="black", padx=10)
probLabel.place(x=40, y=5)
probLabel.configure(text='Probability')

classBox = ck.CTkLabel(window, height=30, width=120, font=("Arial", 14), text_color="white", fg_color="blue")
classBox.place(x=180, y=40)
classBox.configure(text='0')

counterBox = ck.CTkLabel(window, height=30, width=120, font=("Arial", 14), text_color="white", fg_color="blue")
counterBox.place(x=320, y=40)
counterBox.configure(text='0')

probBox = ck.CTkLabel(window, height=30, width=120, font=("Arial", 14), text_color="white", fg_color="blue")
probBox.place(x=40, y=40)
probBox.configure(text='0.00')  # Inicializa con dos decimales

# Frame para el video
frame = tk.Frame(window, height=480, width=460, bg="black") #height=480, width=460
frame.place(x=10, y=80)
lmain = tk.Label(frame)
lmain.place(x=0, y=0)

# Nuevas etiquetas y cajas para el análisis de forma
formLabel = ck.CTkLabel(window, height=30, width=120, font=("Arial", 14), text_color="black", padx=10)
formLabel.place(x=10, y=560)#590
formLabel.configure(text='Form Analysis')

formBox1 = ck.CTkLabel(window, height=30, width=120, font=("Arial", 14), text_color="black", fg_color="orange")
formBox1.place(x=10, y=590) #630
formBox1.configure(text='Narrow')

formProb1 = ck.CTkLabel(window, height=30, width=120, font=("Arial", 14), text_color="black")
formProb1.place(x=160, y=590) #630
formProb1.configure(text='0.00')

formBox2 = ck.CTkLabel(window, height=30, width=120, font=("Arial", 14), text_color="black", fg_color="red")
formBox2.place(x=10, y=630) #670
formBox2.configure(text='Neutral')

formProb2 = ck.CTkLabel(window, height=30, width=120, font=("Arial", 14), text_color="black")
formProb2.place(x=160, y=630)
formProb2.configure(text='0.00')

formAdvice = ck.CTkLabel(window, height=30, width=390, font=("Arial", 14), text_color="black", padx=10, wraplength=420)
formAdvice.place(x=10, y=670) #x=10, y=700
formAdvice.configure(text='')

def reset_counter():
    global counter
    counter = 0

button = ck.CTkButton(window, text='RESET', command=reset_counter, height=30, width=120, font=("Arial", 14),
                      text_color="black", fg_color="blue")
button.place(x=320, y=600)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5)

# Cargar modelos entrenados
with open('proyecto1B.pkl', 'rb') as f:
    model = pickle.load(f)

with open('lean.pkl', 'rb') as f:
    lean_model = pickle.load(f)

with open('hips.pkl', 'rb') as f:
    hips_model = pickle.load(f)

cap = cv2.VideoCapture(0)
current_stage = ''
counter = 0
bodylang_prob = np.array([0, 0])
bodylang_class = ''
form_analysis = {'hips': 'neutral', 'lean': 'neutral'}

def detect():
    global current_stage
    global counter
    global bodylang_class
    global bodylang_prob
    global form_analysis

    # Inicializar variables por defecto
    hips_class = 'neutral'
    lean_class = 'neutral'
    hips_prob = np.array([0.0, 0.0, 0.0])
    lean_prob = np.array([0.0, 0.0, 0.0])
    advice_text = ''
    
    ret, frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))
    
    try: 
        row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
        X = pd.DataFrame([row], columns=landmarks)
        bodylang_prob = model.predict_proba(X)[0]
        bodylang_class = model.predict(X)[0]
        
        # Actualización de la lógica de estado
        if bodylang_class == "down" and bodylang_prob[bodylang_prob.argmax()] > 0.7:
            current_stage = "down" 
        elif current_stage == "down" and bodylang_class == "up" and bodylang_prob[bodylang_prob.argmax()] > 0.7:
            current_stage = "up"
            counter += 1
        
        # Predicción del análisis de forma usando los modelos entrenados
        lean_prob = np.array(lean_model.predict_proba(X)[0])
        lean_class = lean_model.predict(X)[0]
        hips_prob = np.array(hips_model.predict_proba(X)[0])
        hips_class = hips_model.predict(X)[0]
        
        form_analysis = {'lean': lean_class, 'hips': hips_class}
        
        # Determinar el mensaje de consejo
        if hips_class == "narrow" and lean_class == "right":
            advice_text = "Necesitas ampliar tu postura para arreglarla e inclinarte\n hacia la izquierda para enderezar los hombros."
        elif hips_class == "narrow" and lean_class == "neutral":
            advice_text = "Necesitas ampliar tu postura para arreglarla y \nnada con tu inclinación, estás parejo."
        elif hips_class == "narrow" and lean_class == "left":
            advice_text = "Necesitas ampliar tu postura para arreglarla e inclinarte\n hacia la izquierda para enderezar los hombros."
        elif hips_class == "neutral" and lean_class == "right":
            advice_text = "No necesitas hacer nada para arreglar tu postura PERO inclinarse hacia\n la izquierda para enderezar los hombros."
        elif hips_class == "neutral" and lean_class == "neutral":
            advice_text = "No necesitas hacer nada para arreglar tu postura y\nnada con tu inclinación, estás parejo."
        elif hips_class == "neutral" and lean_class == "left":
            advice_text = "No necesitas hacer nada para arreglar tu postura PERO inclinarse\n hacia la derecha para enderezar los hombros."
        elif hips_class == "wide" and lean_class == "right":
            advice_text = "Necesitas juntar los pies para fijar la postura e inclinarte\n hacia la izquierda para enderezar los hombros."
        elif hips_class == "wide" and lean_class == "neutral":
            advice_text = "Necesitas juntar los pies para fijar la postura y\n nada con tu inclinación, estás parejo."
        elif hips_class == "wide" and lean_class == "left":
            advice_text = "Necesitas juntar los pies para fijar la postura e\n inclinarte hacia la derecha para enderezar los hombros."
        
    except Exception as e:
        pass
    
    img = image[:, :460, :]
    imgarr = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(imgarr)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, detect)
 
    counterBox.configure(text=counter)
    # Formatear la probabilidad con dos decimales
    probBox.configure(text='%.2f' % bodylang_prob[bodylang_prob.argmax()]) 
    classBox.configure(text=current_stage)
    
    # Actualizar las cajas de análisis de forma y colores
    formBox1.configure(text=hips_class)
    formProb1.configure(text='%.2f' % hips_prob[hips_prob.argmax()])
    formBox2.configure(text=lean_class)
    formProb2.configure(text='%.2f' % lean_prob[lean_prob.argmax()])
    formAdvice.configure(text=advice_text)
    
    # Cambiar colores de formBox1 según hips_class
    if hips_class == "narrow":
        formBox1.configure(fg_color="red")
    elif hips_class == "neutral":
        formBox1.configure(fg_color="green")
    elif hips_class == "wide":
        formBox1.configure(fg_color="orange")
    
    # Cambiar colores de formBox2 según lean_class
    if lean_class == "left":
        formBox2.configure(fg_color="red")
    elif lean_class == "neutral":
        formBox2.configure(fg_color="green")
    elif lean_class == "right":
        formBox2.configure(fg_color="orange")

detect()
window.mainloop()
