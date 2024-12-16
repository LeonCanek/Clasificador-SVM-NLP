from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import librosa
import numpy as np
import os
import time
import whisper
import difflib
import logging
import joblib
from typing import List

# Configuración básica del logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI()

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite solicitudes desde tu frontend Angular
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permite todos los encabezados
)

UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Cargar el modelo y el escalador
model = joblib.load('parkinson_model.pkl')
scaler = joblib.load('scaler.pkl')

def analyze_semantics(file_path):
    try:
        model = whisper.load_model("base")
        result = model.transcribe(file_path, word_timestamps=True)
        texto_transcrito = result.get("text", "")

        logging.info(f"Texto transcrito: {texto_transcrito}")

        logging.info("Calculando similitud semántica...")
        texto_objetivo = "Juan se rompió una pierna cuando iba en la moto"
        similitud = difflib.SequenceMatcher(None, texto_objetivo.lower(), texto_transcrito.lower()).ratio()

        def clasificar_por_similitud(similitud):
            if similitud < 0.73:
                return "Altamente probable"
            elif 0.73 <= similitud <= 0.75:
                return "Probable"
            else:
                return "Poco probable"

        clasificacion = clasificar_por_similitud(similitud)
        logging.info(f"Clasificación: {clasificacion}")

        return texto_transcrito, similitud, clasificacion

    except Exception as e:
        logging.error(f"Error durante el análisis semántico: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def extract_audio_features(file_path):
    """Extrae las características acústicas del audio."""
    try:
        logging.info("Cargando archivo de audio...")
        y, sr = librosa.load(file_path, sr=None)

        logging.info("Calculando frecuencias fundamentales...")
        pitches, _ = librosa.piptrack(y=y, sr=sr)
        pitches = pitches[pitches > 0]

        if pitches.size == 0:
            raise ValueError("No se encontraron frecuencias válidas en el audio.")

        meanF0 = np.mean(pitches)
        abs_diffs = np.abs(np.diff(1 / pitches))
        jitter = (np.mean(abs_diffs) / (1 / meanF0)) * 100
        jitter_abs = np.mean(abs_diffs)

        rms = librosa.feature.rms(y=y)[0]
        shimmer = np.std(rms) / np.mean(rms) if np.mean(rms) else 0

        # Escalar las características y predecir
        try:
            features = scaler.transform([[meanF0, jitter, jitter_abs, shimmer]])
            prediction = model.predict(features)[0]
        except Exception as e:
            logging.error(f"Error en escalado o predicción: {e}")
            prediction = None

        return meanF0, jitter, jitter_abs, shimmer, prediction

    except Exception as e:
        logging.error(f"Error durante la extracción de características: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/")
async def audios_analizar(audios: List[UploadFile] = File(...)):

    resultados = []
    for audio in audios:

    # Guardar el archivo temporalmente
        file_path = os.path.join(UPLOAD_FOLDER, audio.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await audio.read())

        try:
            # Procesar el audio: Análisis semántico
            logging.info("Iniciando análisi semántico...")
            start_time_semantic = time.time()
            texto_transcrito, similitud, clasificacion = analyze_semantics(file_path)
            semantic_execution_time = time.time() - start_time_semantic

            # Procesar el audio: Extracción de características
            logging.info("Iniciando extracción de características...")
            start_time_features = time.time()
            meanF0, jitter, jitter_abs, shimmer, prediction = extract_audio_features(file_path)
            features_execution_time = time.time() - start_time_features

            # Tiempo total de ejecución
            execution_time = semantic_execution_time + features_execution_time
            resultados.append({
            "filename": audio.filename,
            "texto_transcrito": texto_transcrito,
            "similitud": float(similitud),
            "clasificacion": clasificacion,
            "semantic_execution_time": semantic_execution_time,
            "meanF0Hz": float(meanF0),
            "jitter": float(jitter),
            "jitter_abs": float(jitter_abs),
            "shimmer": float(shimmer),
            "parkinson_prediction": "Positivo" if prediction == 1 else "Negativo",
            "features_execution_time": features_execution_time,
            "execution_time": execution_time
            })
        finally:
            # Eliminar el archivo temporal
            os.remove(file_path)

        # Respuesta JSON plana con las métricas adicionales
    return JSONResponse(content={"resultados": resultados})


