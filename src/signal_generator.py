from typing import Optional
import numpy as np

class SignalGenerator:
    def __init__(self, timeVector: np.ndarray):
        self.timeVector = timeVector
        self.noisySignal: Optional[np.ndarray] = None

    def generateNoisySignal(self, frequency: float = 10, noiseStdDev: float = 0.5) -> np.ndarray:
        """
        Generate a noisy signal using the defined time vector.
        :param frequency: Frequency of the sine wave (default is 10 Hz).
        :param noiseStdDev: Standard deviation of the noise (default is 0.5).
        """
        self.noisySignal = np.sin(2 * np.pi * frequency * self.timeVector) + noiseStdDev * np.random.randn(len(self.timeVector))
        return self.noisySignal

    def importNoisySignal(self, signal: np.ndarray) -> None:
        self.noisySignal = signal

    def getSignal(self) -> Optional[np.ndarray]:
        return self.noisySignal
