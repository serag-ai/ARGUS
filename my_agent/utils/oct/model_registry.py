import os


os.environ.setdefault("KERAS_BACKEND", "torch")

import keras.models as models

_MODEL_CACHE = {}  # key: (model_path,) -> loaded model

def get_oct_model(model_path: str):
    key = (model_path,)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    loaded = models.load_model(model_path)
    _MODEL_CACHE[key] = loaded
    return loaded