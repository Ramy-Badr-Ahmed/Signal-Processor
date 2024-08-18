from scipy import optimize
import numpy as np
from typing import Optional

class SignalFitter:
    def __init__(self, timeVector: np.ndarray, filteredSignal: np.ndarray):
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

    def fitDampedSineWave(self) -> np.ndarray:
        """
        Fit a damped sine wave to the filtered signal using nonlinear least squares.
        default sine wave parameters: amplitudeParam = 1.0, frequencyParam = 10.0, phaseParam = 0.0, decayRateParam = 0.1
        change with setDampedSineWaveParameters().
        """
        def dampedSineFunc(time: np.ndarray, amplitude: float, frequency: float, phase: float, decayRate: float) -> np.ndarray:
            return amplitude * np.exp(-decayRate * time) * np.sin(2 * np.pi * frequency * time + phase)

        def residualsDamped(params: np.ndarray, time: np.ndarray, data: np.ndarray) -> np.ndarray:
            return dampedSineFunc(time, *params) - data

        initialParams = np.array([self.amplitudeParam, self.frequencyParam, self.phaseParam, self.decayRateParam])

        try:
            result = optimize.least_squares(residualsDamped, initialParams, bounds = self.bounds, args = (self.timeVector, self.filteredSignal))
            self.optimalParamsDamped = result.x
        except Exception as e:
            raise RuntimeError(f"Optimization failed: {e}")

        self.fittedSignalDamped = dampedSineFunc(self.timeVector, *self.optimalParamsDamped)

        print("Damped sine wave fitted.")

        return self.fittedSignalDamped

    def getFittedSignal(self) -> Optional[np.ndarray]:
        return self.fittedSignalDamped

    def getOptimalParams(self) -> Optional[np.ndarray]:
        return self.optimalParamsDamped
