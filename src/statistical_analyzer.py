from scipy import stats
import numpy as np
from typing import Optional

class StatisticalAnalyzer:
    def __init__(self, filteredSignal: np.ndarray, fittedSignalDamped: np.ndarray):
        self.filteredSignal = filteredSignal
        self.fittedSignalDamped = fittedSignalDamped
        self.tStatistic: Optional[float] = None
        self.pValue: Optional[float] = None

    def performTTest(self) -> tuple[float, float]:
        """
        Perform a t-test between the filtered signal and the fitted damped sine wave.
        """
        if self.filteredSignal is None or self.fittedSignalDamped is None:
            raise ValueError("Both filtered signal and fitted signal must be available. Please run 'applyFilter' and 'fitDampedSineWave' first.")

        try:
            self.timeStatistic, self.pValue = stats.ttest_ind(self.filteredSignal, self.fittedSignalDamped)
        except Exception as e:
            raise RuntimeError(f"T-test failed: {e}")

        print("T-test performed.")

        return self.tStatistic, self.pValue

    def getTTestResults(self) -> tuple[Optional[float], Optional[float]]:
        return self.tStatistic, self.pValue
