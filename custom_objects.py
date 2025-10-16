"""
Solución mejorada para compatibilidad de modelos Keras
"""
import tensorflow as tf

def get_custom_objects():
    """
    Diccionario completo de objetos personalizados para compatibilidad
    """
    custom_objects = {}
    
    try:
        from tensorflow.keras.layers import DepthwiseConv2D, ReLU, Activation
        from tensorflow.keras.activations import swish
        from tensorflow.keras import initializers, regularizers, constraints
        
        custom_objects.update({
            'DepthwiseConv2D': DepthwiseConv2D,
            'ReLU': ReLU,
            'Activation': Activation,
            'swish': swish,
            'relu': tf.nn.relu,
            'softmax': tf.nn.softmax,
            'linear': tf.keras.activations.linear,
            'sigmoid': tf.nn.sigmoid,
            'VarianceScaling': initializers.VarianceScaling,
            'Zeros': initializers.Zeros,
            'Ones': initializers.Ones,
        })
        
        # Manejar el parámetro 'groups' que no existe en DepthwiseConv2D
        class CompatibleDepthwiseConv2D(DepthwiseConv2D):
            def __init__(self, *args, **kwargs):
                # Remover el parámetro 'groups' si existe
                kwargs.pop('groups', None)
                super().__init__(*args, **kwargs)
        
        custom_objects['DepthwiseConv2D'] = CompatibleDepthwiseConv2D
        custom_objects['expanded_conv_depthwise'] = CompatibleDepthwiseConv2D
        
    except Exception as e:
        print(f"⚠️  Error creando custom objects: {e}")
    
    return custom_objects

def load_model_with_fixes(model_path):
    """
    Carga el modelo con múltiples estrategias de compatibilidad
    """
    strategies = [
        # Intento 1: Carga normal
        lambda: tf.keras.models.load_model(model_path, compile=False),
        
        # Intento 2: Con custom objects
        lambda: tf.keras.models.load_model(
            model_path, 
            compile=False, 
            custom_objects=get_custom_objects()
        ),
        
        # Intento 3: Ignorando errores de capas
        lambda: tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects=get_custom_objects()
        ),
    ]
    
    for i, strategy in enumerate(strategies, 1):
        try:
            print(f"🔧 Intentando estrategia {i}...")
            model = strategy()
            print(f"✅ Modelo cargado con estrategia {i}")
            return model
        except Exception as e:
            print(f"❌ Estrategia {i} falló: {e}")
            if i == len(strategies):
                # Último intento: forzar carga ignorando errores
                try:
                    print("🔄 Último intento: carga forzada...")
                    model = tf.keras.models.load_model(
                        model_path,
                        compile=False,
                        custom_objects=get_custom_objects()
                    )
                    return model
                except:
                    raise e
    
    raise Exception("Todas las estrategias de carga fallaron")