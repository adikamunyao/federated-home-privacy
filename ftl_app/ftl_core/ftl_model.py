# ftl_app/ftl_core/ftl_model.py
import tensorflow as tf
import tensorflow_privacy as tfp
import numpy as np
import json
import os
from sklearn.feature_selection import mutual_info_classif
from .data_generator import generate_synthetic_data, preprocess_data
import logging
import dp_accounting  # Standalone library import

logging.basicConfig(level=logging.INFO)

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

def compute_mutual_information(raw_data, model_updates):
    raw_flat = raw_data.reshape(-1)
    updates_flat = np.concatenate([u.flatten() for u in model_updates])[:len(raw_flat)]
    bins = 10
    raw_binned = np.digitize(raw_flat, np.linspace(raw_flat.min(), raw_flat.max(), bins))
    updates_binned = np.digitize(updates_flat, np.linspace(updates_flat.min(), updates_flat.max(), bins))
    mi = mutual_info_classif(raw_binned.reshape(-1, 1), updates_binned, discrete_features=True)[0]
    entropy_raw = -np.sum([p * np.log2(p) for p in np.bincount(raw_binned) / len(raw_binned) if p > 0])
    return mi / entropy_raw if entropy_raw > 0 else 0

def aggregate_weights(local_weights):
    """Manual FedAvg: Average weights across clients."""
    return [np.mean([w[i] for w in local_weights], axis=0) for i in range(len(local_weights[0]))]

def run_ftl_simulation(num_rounds=5, num_devices=3, noise_multiplier=1.1, precomputed=False):
    if precomputed:
        with open('ftl_app/precomputed/metrics.json', 'r') as f:
            metrics = json.load(f)
        return (metrics['accuracy'], metrics['loss'], metrics['leakage'], 
                metrics['latency'], metrics['epsilon'])

    # Generate and preprocess data
    raw_data = generate_synthetic_data(num_devices)
    client_data = preprocess_data(raw_data)
    logging.info(f"Generated data for {num_devices} devices")

    # Initialize global model with DP optimizer
    global_model = create_local_model()
    optimizer = tfp.DPKerasSGDOptimizer(
        l2_norm_clip=1.0,
        noise_multiplier=noise_multiplier,
        num_microbatches=1,
        learning_rate=0.01
    )
    global_model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )
    logging.info("Global model compiled with DPKerasSGDOptimizer")

    accuracy, loss, leakage, latency, epsilon_values = [], [], [], [], []
    batch_size = 32
    total_examples = num_devices * 200  # n = num_devices * samples_per_device

    for round_num in range(num_rounds):
        logging.info(f"Starting round {round_num + 1}")
        local_weights = []
        local_losses = []
        local_accs = []

        # Simulate local training on each device
        for i, client_dataset in enumerate(client_data):
            local_model = create_local_model()
            local_model.set_weights(global_model.get_weights())
            local_optimizer = tfp.DPKerasSGDOptimizer(
                l2_norm_clip=1.0,
                noise_multiplier=noise_multiplier,
                num_microbatches=1,
                learning_rate=0.01
            )
            local_model.compile(
                optimizer=local_optimizer,
                loss='binary_crossentropy',
                metrics=['binary_accuracy']
            )
            history = local_model.fit(client_dataset, epochs=1, verbose=0)
            local_weights.append(local_model.get_weights())
            local_losses.append(history.history['loss'][0])
            local_accs.append(history.history['binary_accuracy'][0])

        # Aggregate weights manually (FedAvg)
        global_weights = aggregate_weights(local_weights)
        global_model.set_weights(global_weights)

        # Compute metrics
        acc = np.mean(local_accs)
        lss = np.mean(local_losses)
        updates = [g - l for g, l in zip(global_weights, local_weights[0])]
        leak = compute_mutual_information(raw_data[0][0], updates)
        lat = np.random.uniform(0.1, 0.5) * num_devices
        try:
            # Epsilon via dp-accounting standalone library
            accountant = dp_accounting.rdp.RdpAccountant()
            steps = (total_examples // batch_size) * (round_num + 1)  # Total steps up to this round
            event = dp_accounting.GaussianDpEvent(noise_multiplier=noise_multiplier)
            accountant.compose(dp_accounting.SelfComposedDpEvent(event, steps))
            epsilon = accountant.get_epsilon(target_delta=1e-5)
        except Exception as e:
            logging.warning(f"Epsilon calculation failed: {e}")
            epsilon = float('inf')

        accuracy.append(float(acc))
        loss.append(float(lss))
        leakage.append(float(leak))
        latency.append(float(lat))
        epsilon_values.append(float(epsilon))

    # Save weights
    os.makedirs('ftl_app/precomputed/weights', exist_ok=True)
    global_model.save_weights('ftl_app/precomputed/weights/global_model.h5')

    return accuracy, loss, leakage, latency, epsilon_values