import tensorflow as tf
import tensorflow_federated as tff
import tensorflow_privacy as tfp
import numpy as np
import json
import os
from sklearn.feature_selection import mutual_info_classif
from .data_generator import generate_synthetic_data, preprocess_data

def create_local_model(input_dim=2):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def model_fn():
    keras_model = create_local_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=(tf.TensorSpec(shape=[None, 2], dtype=tf.float32),
                    tf.TensorSpec(shape=[None], dtype=tf.int32)),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

def compute_mutual_information(raw_data, model_updates):
    """Calculate mutual information between raw data and model updates."""
    # Flatten raw data and updates for MI calculation
    raw_flat = raw_data.reshape(-1)  # Features flattened
    updates_flat = np.concatenate([u.flatten() for u in model_updates])[:len(raw_flat)]  # Match length
    
    # Discretize continuous data (binning)
    bins = 10
    raw_binned = np.digitize(raw_flat, np.linspace(raw_flat.min(), raw_flat.max(), bins))
    updates_binned = np.digitize(updates_flat, np.linspace(updates_flat.min(), updates_flat.max(), bins))
    
    # Compute MI (normalized between 0 and 1)
    mi = mutual_info_classif(raw_binned.reshape(-1, 1), updates_binned, discrete_features=True)[0]
    entropy_raw = -np.sum([p * np.log2(p) for p in np.bincount(raw_binned) / len(raw_binned) if p > 0])
    return mi / entropy_raw if entropy_raw > 0 else 0  # Normalize by entropy of raw data

def run_ftl_simulation(num_rounds=5, num_devices=3, noise_multiplier=1.1, precomputed=False):
    if precomputed:
        with open('ftl_app/precomputed/metrics.json', 'r') as f:
            metrics = json.load(f)
        return (metrics['accuracy'], metrics['loss'], metrics['leakage'], 
                metrics['latency'], metrics['epsilon'])

    # Generate raw data
    raw_data = generate_synthetic_data(num_devices)
    client_data = preprocess_data(raw_data)

    # Setup FTL
    iterative_process = tff.learning.build_federated_averaging_process(
        model_fn,
        client_optimizer_fn=lambda: tfp.DPGradientDescentGaussianOptimizer(
            l2_norm_clip=1.0, noise_multiplier=noise_multiplier, num_microbatches=1, learning_rate=0.01
        ),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0)
    )
    state = iterative_process.initialize()

    accuracy, loss, leakage, latency, epsilon_values = [], [], [], [], []

    for round_num in range(num_rounds):
        # Simulate local training and get updates
        state, metrics = iterative_process.next(state, client_data)
        acc = metrics['train']['binary_accuracy']
        lss = metrics['train']['loss']
        
        # Extract model updates (simplified: use weights difference)
        global_model = create_local_model()
        # Note: TFF state extraction is complex; this is an approximation
        global_model.set_weights(tff.learning.ModelWeights.from_tff_result(state.model).trainable)
        local_model = create_local_model()
        local_model.fit(client_data[0], epochs=1, verbose=0)  # Train on one client briefly
        updates = [g - l for g, l in zip(global_model.get_weights(), local_model.get_weights())]
        
        # Compute real leakage
        raw_features = raw_data[0][0]  # Use first device's features
        leak = compute_mutual_information(raw_features, updates)
        
        # Latency (simulated)
        lat = np.random.uniform(0.1, 0.5) * num_devices
        
        # Epsilon (placeholder for now; updated in Step 2)
        epsilon = noise_multiplier * np.sqrt(2 * num_rounds)
        
        accuracy.append(float(acc))
        loss.append(float(lss))
        leakage.append(float(leak))
        latency.append(float(lat))
        epsilon_values.append(float(epsilon))

    # Save weights
    os.makedirs('ftl_app/precomputed/weights', exist_ok=True)
    global_model.save_weights('ftl_app/precomputed/weights/global_model.h5')

    return accuracy, loss, leakage, latency, epsilon_values