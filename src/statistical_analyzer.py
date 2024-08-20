from scipy import stats
import numpy as np
from typing import Optional

class StatisticalAnalyzer:
    def __init__(self, filteredSignal: np.ndarray, fittedSignalDamped: np.ndarray):
        """
        Initialize the StatisticalAnalyzer with the original and fitted signals.
        :param filteredSignal: The original filtered signal.
        :param fittedSignalDamped: The fitted signal to compare.
        """
        self.filteredSignal = filteredSignal
        self.fittedSignalDamped = fittedSignalDamped
        self.timeStatistic: Optional[float] = None
        self.pValue: Optional[float] = None

        self.performTTest()

    def performTTest(self) -> 'StatisticalAnalyzer':
        """
        Perform a t-test comparing the filtered signal with the fitted signal.
        """
        if self.fittedSignalDamped is None or len(self.fittedSignalDamped) == 0:
            raise ValueError("Fitted signal is None or empty. Cannot perform t-test.")
        try:
            self.timeStatistic, self.pValue = stats.ttest_ind(self.filteredSignal, self.fittedSignalDamped)
        except Exception as e:
            raise RuntimeError(f"T-test failed: {e}")

        print(f"T-test performed. T-statistic: {self.timeStatistic:.2f}, p-value: {self.pValue:.2e}")

        return self

    def getTTestResults(self) -> tuple[Optional[float], Optional[float]]:
        return self.timeStatistic, self.pValue
