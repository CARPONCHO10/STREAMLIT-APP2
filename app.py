import time
import numpy as np
import cv2
import streamlit as st
import pandas as pd
import sqlite3
import os
import sys
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoTransformerBase

# Configuración de la página
st.set_page_config(
    page_title="Clasificador en vivo", 
    page_icon="🎥", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎥 Clasificación en vivo con Keras + Streamlit")
st.caption("Cámara dentro de la página y resultados en la misma interfaz. Incluye selector de cámara/calidad y registro a CSV.")

# Constantes
MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"
DB_PATH = "predicciones.db"

# =============================================================================
# 1. VERIFICACIÓN INICIAL DE ARCHIVOS Y DEPENDENCIAS
# =============================================================================

def verificar_archivos():
    """Verifica que todos los archivos necesarios existan"""
    st.sidebar.header("📁 Verificación de archivos")
    
    archivos_requeridos = {
        "Modelo Keras": MODEL_PATH,
        "Etiquetas": LABELS_PATH,
        "Requirements": "requirements.txt",
        "Runtime": "runtime.txt"
    }
    
    todos_existen = True
    
    for nombre, archivo in archivos_requeridos.items():
        if os.path.exists(archivo):
            st.sidebar.success(f"✅ {nombre}: {archivo}")
        else:
            st.sidebar.error(f"❌ {nombre}: {archivo} - NO ENCONTRADO")
            todos_existen = False
    
    # Listar todos los archivos en el directorio
    st.sidebar.write("---")
    st.sidebar.write("**Todos los archivos en el directorio:**")
    try:
        files = os.listdir(".")
        for file in sorted(files):
            icon = "📄" if os.path.isfile(file) else "📁"
            st.sidebar.write(f"{icon} {file}")
    except Exception as e:
        st.sidebar.error(f"Error al listar archivos: {e}")
    
    return todos_existen

# Ejecutar verificación
archivos_ok = verificar_archivos()

if not archivos_ok:
    st.error("""
    **❌ Faltan archivos esenciales**
    
    **Solución:**
    1. Asegúrate de que todos los archivos estén en tu repositorio de GitHub
    2. Verifica los nombres exactos (mayúsculas/minúsculas)
    3. Los archivos deben estar en la raíz del proyecto
    """)
    st.stop()

# =============================================================================
# 2. CONFIGURACIÓN DE BASE DE DATOS
# =============================================================================

def init_db():
    """Inicializa la base de datos SQLite"""
    try:
        conn = sqlite3.connect(DB_PATH)
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
        st.sidebar.success("✅ Base de datos inicializada")
    except Exception as e:
        st.sidebar.error(f"❌ Error en BD: {e}")

def save_prediction_to_db(timestamp, label, confidence):
    """Guarda una predicción en la base de datos"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO predicciones (timestamp, label, confidence) VALUES (?, ?, ?)",
                  (timestamp, label, confidence))
        conn.commit()
        conn.close()
    except Exception as e:
        st.sidebar.error(f"Error guardando en BD: {e}")

# Inicializar BD
init_db()

# =============================================================================
# 3. CARGA SEGURA DEL MODELO Y ETIQUETAS
# =============================================================================

@st.cache_resource
def load_model_cached(model_path: str):
    """
    Carga el modelo Keras con manejo de errores y compatibilidad
    """
    try:
        # Importar la función de carga segura
        from custom_objects import load_model_with_fixes
        model = load_model_with_fixes(model_path)
        return model
    except Exception as e:
        st.error(f"""
        **❌ Error crítico al cargar el modelo**
        
        **Detalle del error:** {e}
        
        **Posibles soluciones:**
        1. Verifica que el modelo sea compatible con TensorFlow 2.13
        2. Revisa que el archivo .h5 no esté corrupto
        3. El modelo podría necesitar ser reentrenado o convertido
        """)
        st.stop()

@st.cache_data
def load_labels(labels_path: str):
    """Carga las etiquetas desde el archivo de texto"""
    try:
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f.readlines()]
        st.sidebar.success(f"✅ {len(labels)} etiquetas cargadas")
        return labels
    except Exception as e:
        st.error(f"Error cargando etiquetas: {e}")
        return ["Clase 0", "Clase 1"]  # Etiquetas por defecto

# Cargar modelo y etiquetas
with st.spinner("🔄 Cargando modelo y etiquetas (esto puede tomar unos segundos)..."):
    try:
        model = load_model_cached(MODEL_PATH)
        labels = load_labels(LABELS_PATH)
        st.success("✅ Modelo y etiquetas cargados correctamente")
    except Exception as e:
        st.error(f"Error durante la carga: {e}")
        st.stop()

# =============================================================================
# 4. CONFIGURACIÓN DE LA INTERFAZ
# =============================================================================

# Sidebar - Configuración de cámara
st.sidebar.header("🎦 Ajustes de cámara")

facing = st.sidebar.selectbox(
    "Tipo de cámara", 
    options=["auto (por defecto)", "user (frontal)", "environment (trasera)"],
    index=0,
    help="Selecciona la cámara a usar. En móviles: 'user'=frontal, 'environment'=trasera"
)

quality = st.sidebar.selectbox(
    "Calidad de video",
    options=["640x480", "1280x720", "1920x1080"],
    index=1,
    help="Resolución del video. Mayor resolución = más precisión pero más uso de recursos"
)

# Procesar configuración de video
w, h = map(int, quality.split("x"))
video_constraints = {"width": w, "height": h}
if facing != "auto (por defecto)":
    video_constraints["facingMode"] = facing.split(" ")[0]

media_constraints = {"video": video_constraints, "audio": False}

# Sidebar - Registro de datos
st.sidebar.header("📊 Registro de predicciones")

enable_log = st.sidebar.checkbox(
    "Habilitar registro (CSV + SQLite)", 
    value=True,
    help="Guarda todas las predicciones para análisis posterior"
)

log_every_n_seconds = st.sidebar.slider(
    "Intervalo de registro (segundos)", 
    0.2, 5.0, 1.0, 0.2,
    help="Cada cuántos segundos se guarda una predicción. Evita guardar demasiados datos similares"
)

# =============================================================================
# 5. INICIALIZACIÓN DE SESIÓN
# =============================================================================

if "pred_log" not in st.session_state:
    st.session_state.pred_log = pd.DataFrame(columns=["timestamp", "label", "confidence"])

if "last_log_ts" not in st.session_state:
    st.session_state.last_log_ts = 0.0

if "total_predictions" not in st.session_state:
    st.session_state.total_predictions = 0

# =============================================================================
# 6. CONFIGURACIÓN WEBRTC
# =============================================================================

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class VideoTransformer(VideoTransformerBase):
    def __init__(self) -> None:
        self.latest = {"class": None, "confidence": 0.0}
        self.model = model
        self.labels = labels
        self.prediction_count = 0

    def transform(self, frame):
        # Convertir frame a array numpy
        img = frame.to_ndarray(format="bgr24")
        
        # Preprocesamiento para el modelo
        resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        x = resized.astype(np.float32).reshape(1, 224, 224, 3)
        x = (x / 127.5) - 1.0  # Normalizar a [-1, 1]

        # Realizar predicción
        pred = self.model.predict(x, verbose=0)
        idx = int(np.argmax(pred))
        label = self.labels[idx] if idx < len(self.labels) else f"Clase {idx}"
        conf = float(pred[0][idx])

        # Actualizar información más reciente
        self.latest = {"class": label, "confidence": conf}
        self.prediction_count += 1
        st.session_state.total_predictions = self.prediction_count

        # Dibujar overlay en el video
        overlay = img.copy()
        text = f"{label} | {conf*100:.1f}%"
        
        # Fondo semitransparente para el texto
        cv2.rectangle(overlay, (5, 5), (5 + 12*len(text), 50), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # Texto
        cv2.putText(img, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.0, (0, 255, 0), 2, cv2.LINE_AA)
        
        return img

# =============================================================================
# 7. INTERFAZ PRINCIPAL
# =============================================================================

# Layout de dos columnas
left_col, right_col = st.columns([2, 1], gap="large")

with left_col:
    st.subheader("📹 Cámara en vivo")
    
    # Información de estado
    if st.session_state.total_predictions > 0:
        st.info(f"**Estadísticas:** {st.session_state.total_predictions} predicciones realizadas")
    
    # Componente WebRTC
    webrtc_ctx = webrtc_streamer(
        key="keras-live-classification",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints=media_constraints,
        video_transformer_factory=VideoTransformer,
        async_processing=True,
    )
    
    # Información de ayuda
    with st.expander("💡 Consejos de uso"):
        st.write("""
        - **Navegador recomendado:** Chrome o Edge para mejor compatibilidad
        - **Permisos:** Asegúrate de permitir el acceso a la cámara
        - **Cámara frontal/trasera:** En móviles, usa 'user' para frontal y 'environment' para trasera
        - **Rendimiento:** Si hay lag, reduce la calidad del video
        - **Iluminación:** Mejor resultados con buena iluminación
        """)

with right_col:
    st.subheader("📊 Resultados en tiempo real")
    
    # Placeholders para resultados dinámicos
    result_placeholder = st.empty()
    progress_placeholder = st.empty()
    stats_placeholder = st.empty()
    
    # Inicializar placeholders
    result_placeholder.info("⏳ Esperando activación de la cámara...")
    progress_placeholder.progress(0)
    
    # Botón para limpiar registro
    if enable_log and not st.session_state.pred_log.empty:
        if st.button("🧹 Limpiar registro completo", use_container_width=True):
            st.session_state.pred_log = st.session_state.pred_log.iloc[0:0]
            st.session_state.last_log_ts = 0.0
            st.session_state.total_predictions = 0
            st.rerun()
    
    # Descargar CSV
    if not st.session_state.pred_log.empty:
        csv_bytes = st.session_state.pred_log.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Descargar CSV de predicciones",
            data=csv_bytes,
            file_name=f"predicciones_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        
        # Mostrar estadísticas
        st.write("**Resumen del registro:**")
        st.write(f"- Total de registros: {len(st.session_state.pred_log)}")
        if not st.session_state.pred_log.empty:
            avg_confidence = st.session_state.pred_log['confidence'].mean() * 100
            st.write(f"- Confianza promedio: {avg_confidence:.1f}%")

# =============================================================================
# 8. BUCLE PRINCIPAL DE ACTUALIZACIÓN
# =============================================================================

if webrtc_ctx and webrtc_ctx.state.playing:
    # Inicializar transformer si es necesario
    if webrtc_ctx.video_transformer is None:
        webrtc_ctx.video_transformer = VideoTransformer()
    
    # Bucle de actualización
    for _ in range(1000):  # Límite de iteraciones por seguridad
        if not webrtc_ctx.state.playing:
            break
            
        vt = webrtc_ctx.video_transformer
        if vt is not None and vt.latest["class"] is not None:
            cls = vt.latest["class"]
            conf = vt.latest["confidence"]
            
            # Actualizar interfaz
            result_placeholder.markdown(
                f"**🎯 Clase detectada:** `{cls}`\n\n"
                f"**📈 Confianza:** `{conf*100:.2f}%`"
            )
            progress_placeholder.progress(min(max(conf, 0.0), 1.0))
            
            # Registrar predicción si está habilitado
            if enable_log:
                now = time.time()
                if now - st.session_state.last_log_ts >= log_every_n_seconds:
                    timestamp = datetime.utcnow().isoformat()
                    
                    # Agregar al DataFrame de sesión
                    new_row = pd.DataFrame([{
                        "timestamp": timestamp,
                        "label": cls,
                        "confidence": round(conf, 6)
                    }])
                    st.session_state.pred_log = pd.concat(
                        [st.session_state.pred_log, new_row], 
                        ignore_index=True
                    )
                    
                    # Guardar en base de datos
                    save_prediction_to_db(timestamp, cls, round(conf, 6))
                    st.session_state.last_log_ts = now
        
        time.sleep(0.1)  # Controlar frecuencia de actualización

# =============================================================================
# 9. MODO ALTERNATIVO (SIN WEBRTC)
# =============================================================================

st.markdown("---")
with st.expander("🔄 Modo alternativo: Clasificación por imagen"):
    st.write("""
    **Usa este modo si:**
    - Tu navegador no soporta WebRTC
    - Tu red bloquea la transmisión de video
    - Prefieres clasificar imágenes individuales
    """)
    
    uploaded_file = st.file_uploader(
        "Sube una imagen para clasificar",
        type=['jpg', 'jpeg', 'png'],
        help="Formatos soportados: JPG, JPEG, PNG"
    )
    
    if uploaded_file is not None:
        # Procesar imagen
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        # Mostrar imagen original
        st.image(img, channels="BGR", caption="📷 Imagen original", use_column_width=True)
        
        # Realizar predicción
        with st.spinner("🔍 Analizando imagen..."):
            resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            x = resized.astype(np.float32).reshape(1, 224, 224, 3)
            x = (x / 127.5) - 1.0
            
            pred = model.predict(x, verbose=0)
            idx = int(np.argmax(pred))
            label = labels[idx] if idx < len(labels) else f"Clase {idx}"
            conf = float(pred[0][idx])
        
        # Mostrar resultados
        st.success(f"**🎯 Resultado:** {label}")
        st.metric("📊 Confianza", f"{conf*100:.2f}%")
        
        # Registrar si está habilitado
        if enable_log:
            timestamp = datetime.utcnow().isoformat()
            new_row = pd.DataFrame([{
                "timestamp": timestamp,
                "label": label,
                "confidence": round(conf, 6)
            }])
            st.session_state.pred_log = pd.concat(
                [st.session_state.pred_log, new_row], 
                ignore_index=True
            )
            save_prediction_to_db(timestamp, label, round(conf, 6))
            st.success("✅ Predicción guardada en el registro")

# =============================================================================
# 10. PIE DE PÁGINA
# =============================================================================

st.markdown("---")
st.caption(f"Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")