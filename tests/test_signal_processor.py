import unittest
import numpy as np
from src.signal_generator import SignalGenerator
from src.signal_filter import SignalFilter
from src.signal_fitter import SignalFitter
from src.statistical_analyzer import StatisticalAnalyzer

class TestSignalProcessing(unittest.TestCase):
    def setUp(self):
        self.timeVector = np.linspace(0, 1, 1000, endpoint=False)
        self.noisyInstance = SignalGenerator(self.timeVector)

    def test_generate_noisy_signal(self):
        self.noisyInstance.generateNoisySignal()
        noisySignal = self.noisyInstance.getNoisySignal()
        self.assertIsNotNone(noisySignal)
        self.assertEqual(len(noisySignal), len(self.timeVector))
        self.assertTrue(np.issubdtype(noisySignal.dtype, np.number))

    def test_apply_filter(self):
        self.noisyInstance.generateNoisySignal()
        filteredInstance = self.noisyInstance.applyFilter()
        filteredSignal = filteredInstance.getFilteredSignal()
        self.assertIsNotNone(filteredSignal)
        self.assertEqual(len(filteredSignal), len(self.timeVector))
        self.assertTrue(np.issubdtype(filteredSignal.dtype, np.number))

    def test_fit_damped_sine_wave(self):
        self.noisyInstance.generateNoisySignal()
        filteredInstance = self.noisyInstance.applyFilter()
        fittedInstance = filteredInstance.fitDampedSineWave()
        fittedSignal = fittedInstance.getFittedSignal()
        optimalParams = fittedInstance.getOptimalParams()

        self.assertIsNotNone(fittedSignal)
        self.assertIsNotNone(optimalParams)
        self.assertEqual(len(optimalParams), 4)  # Ensure the right number of parameters

        # Check if the values are within a reasonable range
        amplitude, frequency, phase, decay_rate = optimalParams
        self.assertGreater(amplitude, 0, "Amplitude should be greater than 0")
        self.assertGreater(frequency, 0, "Frequency should be greater than 0")
        self.assertGreaterEqual(decay_rate, 0, "Decay rate should be greater than or equal to 0")
        self.assertTrue(-np.pi <= phase <= np.pi, "Phase should be between -π and π")

    def test_perform_t_test(self):
        self.noisyInstance.generateNoisySignal()
        filteredInstance = self.noisyInstance.applyFilter()
        fittedInstance = filteredInstance.fitDampedSineWave()
        analyzedInstance = fittedInstance.analyzeFit()
        tTestResults = analyzedInstance.getTTestResults()

        self.assertIsNotNone(tTestResults)
        self.assertTrue(np.issubdtype(type(tTestResults[0]), np.number))
        self.assertTrue(np.issubdtype(type(tTestResults[1]), np.number))

    def test_generate_noisy_signal_empty_time_vector(self):
        emptyInstance = SignalGenerator(np.array([]))
        with self.assertRaises(ValueError):
            emptyInstance.generateNoisySignal()

    def test_apply_filter_without_noisy_signal(self):
        # Simulate a case where the noisy signal is not generated before filtering
        emptyInstance = SignalGenerator(self.timeVector)
        with self.assertRaises(ValueError):
            emptyInstance.applyFilter()

    def test_perform_t_test_without_fitted_signal(self):
        self.noisyInstance.generateNoisySignal()
        filteredInstance = self.noisyInstance.applyFilter()
        with self.assertRaises(ValueError):
            StatisticalAnalyzer(self.timeVector, np.array([])).performTTest()

    def test_fit_damped_sine_wave_without_filtered_signal(self):
        self.noisyInstance.generateNoisySignal()
        with self.assertRaises(ValueError):
            SignalFitter(self.timeVector, np.array([])).fitDampedSineWave()

if __name__ == '__main__':
    unittest.main()
