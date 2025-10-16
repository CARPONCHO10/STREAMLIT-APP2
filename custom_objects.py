"""
Archivo para manejar objetos personalizados de Keras/TensorFlow
Soluciona el error de deserialización de DepthwiseConv2D
"""

import tensorflow as tf
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras import Model

def get_custom_objects():
    """
    Retorna un diccionario con las capas personalizadas
    que TensorFlow no puede reconocer automáticamente
    """
    custom_objects = {
        'DepthwiseConv2D': DepthwiseConv2D,
    }
    
    return custom_objects

def load_model_with_fixes(model_path):
    """
    Función segura para cargar modelos con compatibilidad mejorada
    """
    try:
        # Intento 1: Carga normal
        model = tf.keras.models.load_model(model_path, compile=False)
        print("✅ Modelo cargado normalmente")
        return model
    except Exception as e:
        print(f"⚠️  Carga normal falló: {e}")
        print("🔧 Intentando con custom_objects...")
        
        try:
            # Intento 2: Con custom objects
            custom_objects = get_custom_objects()
            model = tf.keras.models.load_model(
                model_path, 
                compile=False, 
                custom_objects=custom_objects
            )
            print("✅ Modelo cargado con custom_objects")
            return model
        except Exception as e2:
            print(f"❌ Error crítico con custom_objects: {e2}")
            raise e2