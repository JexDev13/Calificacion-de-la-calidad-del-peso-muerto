# Proyecto de Análisis de Postura

Este proyecto se encarga de realizar análisis de postura utilizando modelos de Machine Learning y Mediapipe para la detección de landmarks. La arquitectura consta de dos componentes principales:

- **Computadora A**: Hostea el servidor FastAPI que maneja las solicitudes POST para la predicción de la postura.
- **Computadora B**: Captura el video de la cámara y envía los frames a la Computadora A para su procesamiento. También maneja la interfaz de usuario para iniciar y reiniciar el análisis.

## Requisitos

Para clonar y trabajar en este proyecto, asegúrate de tener instalados los siguientes requisitos:

- Python 3.8+
- FastAPI
- uvicorn
- numpy
- opencv-python
- mediapipe
- pydantic
- pandas
- pickle

## Clonar el Proyecto

Para clonar este repositorio puedes hacerlo desde la interfaz de github desktop o usar el siguiente comando:

```bash
git clone [https://github.com/tu-usuario/proyecto-analisis-postura.git](https://github.com/JexDev13/Calificacion-de-la-calidad-del-peso-muerto.git)
cd Calificacion-de-la-calidad-del-peso-muerto
```

## Instalación de Dependencias
Instala las dependencias necesarias utilizando pip:

```bash
pip install fastapi uvicorn numpy opencv-python mediapipe pydantic pandas
```

## Arquitectura
La arquitectura del proyecto consta de dos componentes principales:

Servidor FastAPI (Computadora A): Este servidor recibe las imágenes, realiza el análisis de postura y devuelve los resultados.
Interfaz de Usuario (Computadora B): Captura el video de la cámara, muestra los resultados y permite iniciar o reiniciar el análisis.

### Ejecutar el Servidor FastAPI (Computadora A)
Para ejecutar el servidor FastAPI, utiliza el siguiente comando:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Ejecutar la Interfaz de Usuario (Computadora B)
Abre el archivo index.html en tu navegador web. Asegúrate de estar en la misma red local que la Computadora A. La interfaz de usuario se conecta al servidor FastAPI en la dirección localhost:8000.

```bash
python -m http.server 8000
```

------------------------------
## Notas del desarrollador
* Para trabajar en este proyecto no olvides ajustar las ips de las clases que envian la información a la API a tus propias IPs.

```bash
ipconfig
```
