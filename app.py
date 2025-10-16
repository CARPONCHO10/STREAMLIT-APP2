import time
import numpy as np
import cv2
import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoProcessorBase
from tensorflow.keras.models import load_model
import os

st.set_page_config(page_title="Clasificador en vivo", page_icon="🎥", layout="wide")

st.title("🎥 Clasificación en vivo con Keras + Streamlit")
st.caption("Cámara dentro de la página y resultados en la misma interfaz. Incluye selector de cámara/calidad y registro a CSV.")

MODEL_PATH = "keras_Model.h5"
LABELS_PATH = "labels.txt"

# Verificar que los archivos existan
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ No se encuentra el archivo del modelo: {MODEL_PATH}")
    st.stop()

if not os.path.exists(LABELS_PATH):
    st.error(f"❌ No se encuentra el archivo de etiquetas: {LABELS_PATH}")
    st.stop()

@st.cache_resource
def load_model_cached(model_path: str):
    try:
        return load_model(model_path, compile=False)
    except Exception as e:
        st.error(f"Error cargando el modelo: {e}")
        return None

@st.cache_data
def load_labels(labels_path: str):
    try:
        with open(labels_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines()]
    except Exception as e:
        st.error(f"Error cargando las etiquetas: {e}")
        return []

# Cargar recursos
model = load_model_cached(MODEL_PATH, compile=False)
labels = load_labels(LABELS_PATH)

if model is None or not labels:
    st.stop()

# --- Sidebar: opciones de cámara ---
st.sidebar.header("Ajustes de cámara")
facing = st.sidebar.selectbox(
    "Tipo de cámara", 
    options=["user", "environment"],
    index=0
)

quality = st.sidebar.selectbox(
    "Calidad de video",
    options=["640x480", "1280x720", "1920x1080"],
    index=1
)
w, h = map(int, quality.split("x"))

video_constraints = {
    "width": w, 
    "height": h,
    "facingMode": facing
}

media_constraints = {"video": video_constraints, "audio": False}

st.sidebar.header("Registro de predicciones")
enable_log = st.sidebar.checkbox("Habilitar registro", value=True)
log_every_n_seconds = st.sidebar.slider("Intervalo de registro (s)", 0.2, 5.0, 1.0, 0.2)

if "pred_log" not in st.session_state:
    st.session_state.pred_log = pd.DataFrame(columns=["timestamp", "label", "confidence"])
if "last_log_ts" not in st.session_state:
    st.session_state.last_log_ts = 0.0

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class VideoProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.latest = {"class": None, "confidence": 0.0}
        self.model = model
        self.labels = labels

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        x = resized.astype(np.float32).reshape(1, 224, 224, 3)
        x = (x / 127.5) - 1.0

        pred = self.model.predict(x, verbose=0)
        idx = int(np.argmax(pred))
        label = self.labels[idx] if idx < len(self.labels) else f"Clase {idx}"
        conf = float(pred[0][idx])

        self.latest = {"class": label, "confidence": conf}

        # Overlay con resultados
        overlay = img.copy()
        text = f"{label} | {conf*100:.1f}%"
        cv2.rectangle(overlay, (5, 5), (5 + 8*len(text), 45), (0, 0, 0), -1)
        cv2.putText(overlay, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        
        return overlay

left, right = st.columns([2, 1], gap="large")

with left:
    st.subheader("Cámara en vivo")
    webrtc_ctx = webrtc_streamer(
        key="keras-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints=media_constraints,
        video_processor_factory=VideoProcessor,
        async_processing=True,
    )
    
    st.info(
        "Activa la cámara para comenzar la clasificación en tiempo real. "
        "En móviles, 'user' = frontal y 'environment' = trasera.",
        icon="ℹ️",
    )

with right:
    st.subheader("Resultados")
    result_placeholder = st.empty()
    progress_placeholder = st.empty()

    if enable_log and not st.session_state.pred_log.empty:
        if st.button("🧹 Limpiar registro"):
            st.session_state.pred_log = st.session_state.pred_log.iloc[0:0]
            st.session_state.last_log_ts = 0.0

    if not st.session_state.pred_log.empty:
        csv_bytes = st.session_state.pred_log.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Descargar CSV",
            data=csv_bytes,
            file_name=f"predicciones_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

    # Mostrar predicciones en tiempo real
    if webrtc_ctx and webrtc_ctx.state.playing:
        vt = webrtc_ctx.video_processor
        if vt is not None and vt.latest["class"] is not None:
            cls = vt.latest["class"]
            conf = vt.latest["confidence"]
            result_placeholder.markdown(f"**Clase detectada:** `{cls}`\n\n**Confianza:** `{conf*100:.2f}%`")
            progress_placeholder.progress(min(max(conf, 0.0), 1.0))

            if enable_log:
                now = time.time()
                if now - st.session_state.last_log_ts >= log_every_n_seconds:
                    timestamp = datetime.utcnow().isoformat()
                    st.session_state.pred_log.loc[len(st.session_state.pred_log)] = [
                        timestamp,
                        cls,
                        round(conf, 6),
                    ]
                    st.session_state.last_log_ts = now
    else:
        st.write("Activa la cámara para ver las predicciones aquí.")

# Modo alternativo con captura de foto
st.markdown("---")
with st.expander("📸 Modo alternativo (captura por foto)"):
    st.write("Si hay problemas con la cámara en vivo, usa esta opción:")
    snap = st.camera_input("Captura una imagen para clasificar")
    if snap is not None:
        file_bytes = np.asarray(bytearray(snap.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        x = resized.astype(np.float32).reshape(1, 224, 224, 3)
        x = (x / 127.5) - 1.0
        pred = model.predict(x, verbose=0)
        idx = int(np.argmax(pred))
        label = labels[idx] if idx < len(labels) else f"Clase {idx}"
        conf = float(pred[0][idx])
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Imagen capturada", use_column_width=True)
        with col2:
            st.success(f"**Predicción:** {label}")
            st.metric("Confianza", f"{conf*100:.2f}%")
        
        if enable_log:
            timestamp = datetime.utcnow().isoformat()
            st.session_state.pred_log.loc[len(st.session_state.pred_log)] = [
                timestamp,
                label,
                round(conf, 6),
            ]