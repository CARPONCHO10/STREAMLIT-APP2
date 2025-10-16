import time
import numpy as np
import cv2
import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoProcessorBase
import os

st.set_page_config(page_title="Clasificador en vivo", page_icon="🎥", layout="wide")

st.title("🎥 Clasificación en vivo con Streamlit")
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

@st.cache_data
def load_labels(labels_path: str):
    try:
        with open(labels_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines()]
    except Exception as e:
        st.error(f"Error cargando las etiquetas: {e}")
        return []

# Cargar etiquetas
labels = load_labels(LABELS_PATH)

if not labels:
    st.stop()

# --- SQLite: crear tabla si no existe ---
def init_db():
    conn = sqlite3.connect("predicciones.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS predicciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            label TEXT,
            confidence REAL
        )
    """)
    conn.commit()
    conn.close()

def save_prediction_to_db(timestamp, label, confidence):
    conn = sqlite3.connect("predicciones.db")
    c = conn.cursor()
    c.execute("INSERT INTO predicciones (timestamp, label, confidence) VALUES (?, ?, ?)",
              (timestamp, label, confidence))
    conn.commit()
    conn.close()

init_db()

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
        self.latest = {"class": "Demo Mode", "confidence": 0.85}
        self.labels = labels

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Simular predicción (modo demo)
        # En una implementación real, aquí iría la carga y predicción del modelo
        label = self.labels[0] if self.labels else "Demo Class"
        conf = 0.85
        
        self.latest = {"class": label, "confidence": conf}

        # Overlay con resultados
        overlay = img.copy()
        text = f"{label} | {conf*100:.1f}% (Demo)"
        cv2.rectangle(overlay, (5, 5), (5 + 8*len(text), 45), (0, 0, 0), -1)
        cv2.putText(overlay, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        
        return overlay

left, right = st.columns([2, 1], gap="large")

with left:
    st.subheader("Cámara en vivo")
    st.warning("⚠️ Modo Demo: La clasificación está simulada. Para usar el modelo real, necesitarías TensorFlow/Keras.")
    
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
        for _ in range(300000):
            if not webrtc_ctx.state.playing:
                break
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
                        save_prediction_to_db(timestamp, cls, round(conf, 6))
                        st.session_state.last_log_ts = now

            time.sleep(0.2)
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
        
        # Simular predicción
        label = labels[0] if labels else "Demo Class"
        conf = 0.78
        
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
            save_prediction_to_db(timestamp, label, round(conf, 6))