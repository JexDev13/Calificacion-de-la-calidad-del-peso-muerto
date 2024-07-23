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
window.geometry("480x700")
window.title("Proyecto IA")
ck.set_appearance_mode("dark")

classLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", padx=10)
classLabel.place(x=10, y=1)
classLabel.configure(text='STAGE')

counterLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", padx=10)
counterLabel.place(x=160, y=1)
counterLabel.configure(text='REPS')

probLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", padx=10)
probLabel.place(x=300, y=1)
probLabel.configure(text='PROB')

classBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
classBox.place(x=10, y=41)
classBox.configure(text='0')

counterBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
counterBox.place(x=160, y=41)
counterBox.configure(text='0')

probBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
probBox.place(x=300, y=41)
probBox.configure(text='0.00')  # Inicializa con dos decimales

counter = 0

def reset_counter():
    global counter
    counter = 0

button = ck.CTkButton(window, text='RESET', command=reset_counter, height=40, width=120, font=("Arial", 20),
                      text_color="black", fg_color="blue")
button.place(x=10, y=600)

frame = tk.Frame(height=480, width=480)
frame.place(x=10, y=90)
lmain = tk.Label(frame)
lmain.place(x=0, y=0)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5)

with open('lean.pkl', 'rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)
current_stage = ''
bodylang_prob = np.array([0, 0])
bodylang_class = ''

def detect():
    global current_stage
    global counter
    global bodylang_class
    global bodylang_prob
    
    ret, frame = cap.read()
    
    # Verifica si la cámara se abrió correctamente
    if not ret:
        print("No se puede acceder a la cámara")
        return

    # Recolorear el feed
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convertir BGR a RGB
    image.flags.writeable = False

    # Hacer decisiones
    results = pose.process(image)

    # Recolorear la imagen de nuevo a BGR para renderizar
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convertir RGB a BGR

    # Dibujar las landmarks de la pose
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        try:
            # Obtener las landmarks
            row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
                
            if len(row) != 132:
                print(f"Error: La fila generada tiene {len(row)} elementos, se esperaban 132.")
                return
                
            # Asegúrate de definir 'landmarks' y 'model' en tu contexto
            X = pd.DataFrame([row], columns=landmarks)
            bodylang_class = model.predict(X)[0]
            bodylang_prob = model.predict_proba(X)[0]
               
            print(bodylang_class, bodylang_prob)

            if bodylang_class == 'left' and bodylang_prob[np.argmax(bodylang_prob)] >= .7:
                current_stage = 'left'
            elif current_stage == 'left' and bodylang_class == 'right' and bodylang_prob[np.argmax(bodylang_prob)] <= .7:
                current_stage = "right"
                counter += 1
                print(current_stage)

            # Dibujar los cuadros de las etiquetas
            cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)

            cv2.putText(image, 'CLASS', (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, bodylang_class.split(' ')[0], (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Mostrar la probabilidad
            cv2.putText(image, 'PROB', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(bodylang_prob[np.argmax(bodylang_prob)], 2)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        except Exception as e:
            print(f"Error: {e}")
            pass
    
    img = image[:, :460, :]
    imgarr = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convertir BGR a RGB antes de usar PIL
    imgtk = ImageTk.PhotoImage(imgarr)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, detect)
 
    counterBox.configure(text=counter)
    probBox.configure(text='%.2f' % bodylang_prob[bodylang_prob.argmax()]) 
    classBox.configure(text=current_stage)
      
detect()
window.mainloop()
