import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

def generate_synthetic_data(num_devices=3, num_samples=200):
    """Generate diverse, non-IID smart home data."""
    data = []
    device_types = ['thermostat', 'camera', 'lock']
    for i in range(num_devices):
        device_type = device_types[i % len(device_types)]
        if device_type == 'thermostat':
            temp = np.random.normal(loc=20 + i*2, scale=3, size=num_samples)
            energy = np.random.normal(loc=40 + i*5, scale=15, size=num_samples)
            occupancy = np.random.binomial(1, 0.6 + i*0.05, size=num_samples)
            features = np.stack([temp, energy], axis=1)
            labels = occupancy
        elif device_type == 'camera':
            motion = np.random.normal(loc=10 + i*3, scale=5, size=num_samples)
            light = np.random.normal(loc=50 + i*10, scale=20, size=num_samples)
            activity = np.random.binomial(1, 0.5 + i*0.1, size=num_samples)
            features = np.stack([motion, light], axis=1)
            labels = activity
        else:  # lock
            events = np.random.normal(loc=5 + i*2, scale=2, size=num_samples)
            time = np.random.uniform(0, 24, size=num_samples)
            security = np.random.binomial(1, 0.7 + i*0.03, size=num_samples)
            features = np.stack([events, time], axis=1)
            labels = security
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        data.append((features.astype(np.float32), labels.astype(np.int32)))
    return data

def preprocess_data(device_data):
    """Convert to TFF-compatible datasets."""
    return [tf.data.Dataset.from_tensor_slices((f, l)).shuffle(100).batch(32) for f, l in device_data]