from src.statistical_analyzer import StatisticalAnalyzer
from scipy import optimize
import numpy as np
from typing import Optional

class SignalFitter:
    def __init__(self, timeVector: np.ndarray, filteredSignal: np.ndarray):
        """
        Initialize the SignalFitter with a time vector and a filtered signal.
        :param timeVector: The time vector associated with the signal.
        :param filteredSignal: The filtered signal to fit.
        """
        self.timeVector = timeVector
        self.filteredSignal = filteredSignal

        self.optimalParamsDamped: Optional[np.ndarray] = None
        self.fittedSignalDamped: Optional[np.ndarray] = None

        # Default parameters
        self.amplitudeParam: float = 1.0
        self.frequencyParam: float = 10.0
        self.phaseParam: float = 0.0
        self.decayRateParam: float = 0.1

        # Default bounds
        self.bounds: tuple = ([0, 0, -np.pi, 0], [np.inf, np.inf, np.pi, np.inf])

        self.fitDampedSineWave()

    def setDampedSineWaveParameters(self, amplitudeParam: float, frequencyParam: float, phaseParam: float, decayRateParam: float) -> None:
        """
        Set the parameters for the damped sine wave fitting.
        :param amplitudeParam: Amplitude parameter for the sine wave.
        :param frequencyParam: Frequency parameter for the sine wave.
        :param phaseParam: Phase parameter for the sine wave.
        :param decayRateParam: Decay rate parameter for the sine wave.
        """
        self.amplitudeParam = amplitudeParam
        self.frequencyParam = frequencyParam
        self.phaseParam = phaseParam
        self.decayRateParam = decayRateParam

    def setDampedSineWaveBounds(self, lower: list[float], upper: list[float]) -> None:
        """
        Set the bounds for the damped sine wave fitting parameters.
        Default: ([0, 0, -np.pi, 0], [np.inf, np.inf, np.pi, np.inf])
        :param lower: Lower bounds for the parameters.
        :param upper: Upper bounds for the parameters.
        """
        if len(lower) != 4 or len(upper) != 4:
            raise ValueError("Bounds should be lists of length 4.")
        self.bounds = (lower, upper)

    def fitDampedSineWave(self) -> 'SignalFitter':
        """
        Fit a damped sine wave to the filtered signal using nonlinear least squares.
        default sine wave parameters: amplitudeParam = 1.0, frequencyParam = 10.0, phaseParam = 0.0, decayRateParam = 0.1
        change with setDampedSineWaveParameters().
        """
        if self.filteredSignal is None or len(self.filteredSignal) == 0:
            raise ValueError("Filtered signal is None or empty. Cannot fit a damped sine wave.")

        def dampedSineFunc(time: np.ndarray, amplitude: float, frequency: float, phase: float, decayRate: float) -> np.ndarray:
            """Define the damped sine wave function."""
            return amplitude * np.exp(-decayRate * time) * np.sin(2 * np.pi * frequency * time + phase)

        def residualsDamped(params: np.ndarray, time: np.ndarray, data: np.ndarray) -> np.ndarray:
            """Calculate residuals for the damped sine wave fitting."""
            return dampedSineFunc(time, *params) - data

        initialParams = np.array([self.amplitudeParam, self.frequencyParam, self.phaseParam, self.decayRateParam])

        try:
            result = optimize.least_squares(residualsDamped, initialParams, bounds = self.bounds, args = (self.timeVector, self.filteredSignal))
            self.optimalParamsDamped = result.x

        except Exception as e:
            raise RuntimeError(f"Optimization failed: {e}")

        self.fittedSignalDamped = dampedSineFunc(self.timeVector, *self.optimalParamsDamped)
        print("Damped sine wave fitted.")
        return self

    def getFittedSignal(self) -> Optional[np.ndarray]:
        """
        Retrieve the fitted damped sine wave signal.
        :return: The fitted signal or None if fitting hasn't been performed.
        """
        return self.fittedSignalDamped

    def getOptimalParams(self) -> Optional[np.ndarray]:
        """
        Retrieve the optimal parameters obtained from fitting a damped sine wave to the signal.
        returns Parameters: Amplitude (A), Frequency (f), Phase (φ), Decay rate (λ)
        :return: A numpy array containing the optimal parameters or None if fitting hasn't been performed.
        """
        return self.optimalParamsDamped

    def analyzeFit(self):
        """
        Create a StatisticalAnalyzer instance to analyze the fitted signal against the filtered signal.
        :return: A StatisticalAnalyzer instance.
        """
        if self.fittedSignalDamped is None:
            raise ValueError("Fitted signal is not available. Perform fitting first.")
        return StatisticalAnalyzer(self.filteredSignal, self.fittedSignalDamped)
