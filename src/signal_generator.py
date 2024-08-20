from src.signal_filter import SignalFilter
from typing import Optional
import numpy as np

class SignalGenerator:
    def __init__(self, timeVector: np.ndarray):
        """
        Initialize the SignalGenerator with a time vector.
        """
        self.timeVector = timeVector
        self.noisySignal: Optional[np.ndarray] = None

    def generateNoisySignal(self, frequency: float = 10, noiseStdDev: float = 0.5) -> 'SignalGenerator':
        """
        Generate a noisy signal using the defined time vector.
        :param frequency: Frequency of the sine wave (default is 10 Hz).
        :param noiseStdDev: Standard deviation of the noise (default is 0.5).
        """
        if self.timeVector is None or len(self.timeVector) == 0:
            raise ValueError("Time array 'timeVector' is not properly initialised.")

        self.noisySignal = np.sin(2 * np.pi * frequency * self.timeVector) + noiseStdDev * np.random.randn(len(self.timeVector))
        return self

    def importNoisySignal(self, signal: np.ndarray) -> 'SignalGenerator':
        """
        Import an externally generated noisy signal.
        :param signal: The noisy signal to import.
        """
        self.noisySignal = signal
        return self

    def getNoisySignal(self) -> Optional[np.ndarray]:
        """
        Retrieve the current noisy signal.
        :return: The noisy signal or None if it hasn't been set.
        """
        return self.noisySignal

    def applyFilter(self) -> SignalFilter:
        """
        Create a SignalFilter instance to filter the noisy signal.
        :return: A SignalFilter instance.
        """
        if self.noisySignal is None:
            raise ValueError("Noisy signal has not been generated or imported.")
        return SignalFilter(self.timeVector, self.noisySignal)

