# ftl_app/ftl_core/evaluation.py
import numpy as np
import tensorflow as tf
from .data_generator import generate_synthetic_data
from .ftl_model import create_local_model

def simulate_inference_attack(num_devices=3):
    data = generate_synthetic_data(num_devices)
    model = create_local_model()
    model.load_weights('ftl_app/precomputed/weights/global_model.h5')
    success_rate = 0
    for features, labels in data:
        preds = model.predict(features, verbose=0)
        inferred = (preds > 0.5).astype(int).flatten()
        success_rate += np.mean(inferred == labels)
    return success_rate / num_devices