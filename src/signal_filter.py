from scipy import signal
from typing import Optional
import numpy as np

class SignalFilter:
    def __init__(self, noisySignal: np.ndarray, sampleRate: float = 1.0):
        self.sampleRate = sampleRate
        self.noisySignal = noisySignal
        self.filteredSignal: Optional[np.ndarray] = None

        self.filterTypes = {
            "butter": lambda order, cutoff, btype: signal.butter(order, cutoff, btype, analog = False),
            "chebyshev1": lambda order, cutoff, btype: signal.cheby1(order, 0.5, cutoff, btype, analog = False),
            "chebyshev2": lambda order, cutoff, btype: signal.cheby2(order, 20, cutoff, btype, analog = False),
            "elliptic": lambda order, cutoff , btype: signal.ellip(order, 0.5, 20, cutoff,  analog = False),
            "bessel": lambda order, cutoff, btype: signal.bessel(order, cutoff, btype, analog = False),
            "notch": lambda notchFreq, Q: signal.iirnotch(notchFreq, Q, fs = self.sampleRate),
            "highpass": lambda order, cutoff, btype: signal.butter(order, cutoff, btype, analog = False),
            "bandpass": lambda low_cutoff, high_cutoff, btype: signal.butter(4, [low_cutoff, high_cutoff], btype, analog = False),
            "bandstop": lambda low_cutoff, high_cutoff, btype: signal.butter(4, [low_cutoff, high_cutoff], btype, analog = False)
        }

    def apply_filter(self, filterType: str = 'butter', **kwargs) -> np.ndarray:
        """
        Apply the specified filter type.
        :param filterType: Type of filter ('butter', 'chebyshev1', 'chebyshev2', 'elliptic', 'bessel', 'notch', 'highpass', 'bandpass', 'bandstop').
        :param kwargs: Additional parameters for specific filters.
        :return: Filtered signal as a numpy array.
        """
        if filterType not in self.filterTypes:
            raise ValueError(f"Filter type '{filterType}' is not recognized.")

        if self.noisySignal is None:
            raise ValueError("Noisy signal is not generated. Please call 'generateNoisySignal' first.")

        # Design Butterworth low-pass filter
        [filterCoefficientsB, filterCoefficientsA] = self.filterTypes[filterType](**kwargs)

        self.filteredSignal = signal.filtfilt(filterCoefficientsB, filterCoefficientsA, self.noisySignal)

        print("Filter applied.")

        return self.filteredSignal

    def getFilteredSignal(self) -> Optional[np.ndarray]:
        return self.filteredSignal
