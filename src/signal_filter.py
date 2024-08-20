from src.signal_fitter import SignalFitter
from scipy import signal
from typing import Optional
import numpy as np

class SignalFilter:
    def __init__(self, timeVector: np.ndarray, noisySignal: np.ndarray):
        """
        Initialize with a noisy signal to filter.
        Defaults:
        filterOrder: Order of the Butterworth filter (default is 4).
        cutoffFrequency: Cutoff frequency for the low-pass filter (default is 0.2).
        filterType: Type of filter ('butter', 'bessel', 'highpass'). Default is 'butter'.
        :param timeVector: The time vector associated with the signal.
        :param noisySignal: The noisy signal to be filtered.
        """
        self.timeVector = timeVector
        self.noisySignal = noisySignal
        self.filteredSignal: Optional[np.ndarray] = None

        # Default parameters
        self.filterType: str = 'butter'
        self.filterOrder: int = 4
        self.cutOffFrequency: float = 0.2
        self.bType: str = 'low'

        self.filterTypes = {
            "butter": lambda order, cutoff, btype: signal.butter(order, cutoff, btype, analog = False),
            "bessel": lambda order, cutoff, btype: signal.bessel(order, cutoff, btype, analog = False),
            "highpass": lambda order, cutoff, btype: signal.butter(order, cutoff, btype, analog = False)
        }

        self.applyFilter(order = self.filterOrder, cutoff = self.cutOffFrequency, btype = self.bType)

    def setFilterParameters(self, filterType: str, filterOrder: int, cutOffFrequency: float, bType: str) -> None:
        """
        Set or change filter parameters.
        :param filterType: Type of filter ('butter', 'bessel', 'highpass').
        :param filterOrder: Order of the filter.
        :param cutOffFrequency: Cutoff frequency for the filter.
        :param bType: Filter band type ('lowpass', 'highpass', etc.).
        """
        self.filterType = filterType
        self.throwIfNotSupported()
        self.filterOrder = filterOrder
        self.cutOffFrequency = cutOffFrequency
        self.bType = bType

    def throwIfNotSupported(self):
        """
        Raise an error if the filter type is not supported.
        """
        if self.filterType not in self.filterTypes:
            raise ValueError(f"Filter type '{self.filterType}' is not recognized.")

    def applyFilter(self, **kwargs) -> 'SignalFilter':
        """
        Apply the configured filter to the noisy signal.
        :param kwargs: Additional parameters for specific filters. Given as Dict with keys: Order, cutoff, btype
        """
        [filterCoefficientsB, filterCoefficientsA] = self.filterTypes[self.filterType](**kwargs)
        self.filteredSignal = signal.filtfilt(filterCoefficientsB, filterCoefficientsA, self.noisySignal)
        print("Filter applied.")
        return self

    def fitDampedSineWave(self) -> SignalFitter:
        """
        Create a SignalFitter instance to fit a damped sine wave to the filtered signal.
        :return: A SignalFitter instance.
        """
        return SignalFitter(self.timeVector, self.filteredSignal)

    def getFilteredSignal(self) -> Optional[np.ndarray]:
        """
        Retrieve the filtered signal.
        :return: The filtered signal or None if filtering hasn't been performed.
        """
        return self.filteredSignal
