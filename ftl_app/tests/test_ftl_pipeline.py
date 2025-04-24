import unittest
import numpy as np
import tensorflow as tf
from ftl_app.ftl_core import data_generator, ftl_model

class TestFTLPipeline(unittest.TestCase):

    def setUp(self):
        self.num_devices = 3
        self.num_samples = 100
        self.synthetic_data = data_generator.generate_synthetic_data(
            num_devices=self.num_devices, 
            num_samples=self.num_samples
        )

    def test_generate_synthetic_data_output(self):
        self.assertEqual(len(self.synthetic_data), self.num_devices)
        for features, labels in self.synthetic_data:
            self.assertEqual(features.shape[0], self.num_samples)
            self.assertEqual(labels.shape[0], self.num_samples)
            self.assertEqual(features.shape[1], 2)  # Should always be 2 features
            self.assertTrue(features.dtype == np.float32)
            self.assertTrue(labels.dtype == np.int32)

    def test_preprocess_data_structure(self):
        processed = data_generator.preprocess_data(self.synthetic_data)
        self.assertEqual(len(processed), self.num_devices)
        for dataset in processed:
            self.assertIsInstance(dataset, tf.data.Dataset)
            for batch in dataset.take(1):
                x, y = batch
                self.assertEqual(len(x.shape), 2)  # (batch_size, feature_dim)
                self.assertEqual(len(y.shape), 1)  # (batch_size,)

    def test_model_structure(self):
        model = ftl_model.create_local_model()
        self.assertIsInstance(model, tf.keras.Model)
        self.assertEqual(model.input_shape[1], 2)
        self.assertEqual(model.output_shape[-1], 1)
        self.assertEqual(model.layers[-1].activation.__name__, 'sigmoid')

    def test_run_ftl_simulation_outputs(self):
        accuracy, loss, leakage, latency, epsilon = ftl_model.run_ftl_simulation(
            num_rounds=2, num_devices=2, noise_multiplier=0.5
        )
        for metric in [accuracy, loss, leakage, latency, epsilon]:
            self.assertEqual(len(metric), 2)  # Should have one entry per round
            self.assertTrue(all(isinstance(x, float) for x in metric))

    def test_aggregate_weights_consistency(self):
        model = ftl_model.create_local_model()
        weights = [model.get_weights(), model.get_weights()]
        avg_weights = ftl_model.aggregate_weights(weights)
        for w in avg_weights:
            self.assertIsInstance(w, np.ndarray)

    def test_compute_mutual_information_range(self):
        raw, _ = self.synthetic_data[0]
        dummy_updates = [np.random.normal(size=w.shape) for w in ftl_model.create_local_model().get_weights()]
        mi = ftl_model.compute_mutual_information(raw, dummy_updates)
        self.assertTrue(0 <= mi <= 1 or np.isnan(mi))

if __name__ == '__main__':
    unittest.main()
