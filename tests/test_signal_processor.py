import unittest
import numpy as np
from src.signal_processor import SignalProcessor

class TestSignalProcessor(unittest.TestCase):
    def setUp(self):
        self.timeVector = np.linspace(0, 1, 1000, endpoint=False)
        self.processor = SignalProcessor(self.timeVector)

    def test_generate_noisy_signal(self):
        self.processor.generateNoisySignal()
        self.assertIsNotNone(self.processor.noisySignal)
        self.assertEqual(len(self.processor.noisySignal), len(self.timeVector))
        self.assertTrue(np.issubdtype(self.processor.noisySignal.dtype, np.number))

    def test_apply_filter(self):
        self.processor.generateNoisySignal()
        self.processor.applyFilter()
        self.assertIsNotNone(self.processor.filteredSignal)
        self.assertEqual(len(self.processor.filteredSignal), len(self.timeVector))
        self.assertTrue(np.issubdtype(self.processor.filteredSignal.dtype, np.number))

    def test_fit_damped_sine_wave(self):
        self.processor.generateNoisySignal()
        self.processor.applyFilter()
        self.processor.fitDampedSineWave()
        self.assertIsNotNone(self.processor.fittedSignalDamped)
        self.assertIsNotNone(self.processor.optimalParamsDamped)
        self.assertEqual(len(self.processor.optimalParamsDamped), 4)    # Ensure the right number of parameters

        # Check if the values are within a reasonable range
        amplitude, frequency, phase, decay_rate = self.processor.getParameter('optimalParamsDamped')
        self.assertGreater(amplitude, 0, "Amplitude should be greater than 0")
        self.assertGreater(frequency, 0, "Frequency should be greater than 0")
        self.assertGreaterEqual(decay_rate, 0, "Decay rate should be greater than or equal to 0")
        self.assertTrue(-np.pi <= phase <= np.pi, "Phase should be between -π and π")

    def test_perform_t_test(self):
        self.processor.generateNoisySignal()
        self.processor.applyFilter()
        self.processor.fitDampedSineWave()
        self.processor.performTTest()
        self.assertIsNotNone(self.processor.timeStatistic)
        self.assertIsNotNone(self.processor.pValue)
        self.assertTrue(np.issubdtype(type(self.processor.timeStatistic), np.number))
        self.assertTrue(np.issubdtype(type(self.processor.pValue), np.number))

    def test_generate_noisy_signal_empty_time_vector(self):
        empty_processor = SignalProcessor(np.array([]))
        with self.assertRaises(ValueError):
            empty_processor.generateNoisySignal()

    def test_apply_filter_without_noisy_signal(self):
        with self.assertRaises(ValueError):
            self.processor.applyFilter()

    def test_fit_damped_sine_wave_without_filtered_signal(self):
        self.processor.generateNoisySignal()
        with self.assertRaises(ValueError):
            self.processor.fitDampedSineWave()

    def test_perform_t_test_without_fitted_signal(self):
        self.processor.generateNoisySignal()
        self.processor.applyFilter()
        with self.assertRaises(ValueError):
            self.processor.performTTest()

if __name__ == '__main__':
    unittest.main()
