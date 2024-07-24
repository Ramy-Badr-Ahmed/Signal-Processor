import os

import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy import signal, optimize, stats


class SignalProcessor:
    def __init__(self, timeVector):
        """
        Initialize the SignalProcessor with a time vector.
        :param timeVector: A numpy array of time points at which signals are sampled.
        """
        self.timeVector = timeVector
        self.noisySignal = None
        self.filteredSignal = None
        self.fittedSignalDamped = None
        self.optimalParamsDamped = None
        self.timeStatistic = None
        self.pValue = None

        # Default parameters for the damped sine wave
        self.amplitudeParam = 1.0
        self.frequencyParam = 10.0
        self.phaseParam = 0.0
        self.decayRateParam = 0.1

        # Default bounds for the damped sine wave
        self.bounds = ([0, 0, -np.pi, 0], [np.inf, np.inf, np.pi, np.inf])

    def generateNoisySignal(self, frequency = 10, noiseStdDev = 0.5):
        """
        Generate a noisy signal using the defined time vector.
        :param frequency: Frequency of the sine wave (default is 10 Hz).
        :param noiseStdDev: Standard deviation of the noise (default is 0.5).
        """
        if self.timeVector is None or len(self.timeVector) == 0:
            raise ValueError("Time array 'timeVector' is not properly initialised.")

        self.noisySignal = np.sin(2 * np.pi * frequency * self.timeVector) + noiseStdDev * np.random.randn(len(self.timeVector))
        print("Noisy signal generated.")

    def getParameter(self, paramName):
        """
        Get the value of a specified parameter.
        :param paramName: The name of the parameter to retrieve.
        :return: The value of the specified parameter or None if not found.
        """
        paramMap = {
            'timeVector': self.timeVector,
            'noisySignal': self.noisySignal,
            'filteredSignal': self.filteredSignal,
            'fittedSignalDamped': self.fittedSignalDamped,
            'optimalParamsDamped': self.optimalParamsDamped,
            'timeStatistic': self.timeStatistic,
            'pValue': self.pValue
        }

        if paramName in paramMap:
            return paramMap[paramName]
        else:
            raise ValueError(f"Parameter '{paramName}' does not exist. Available parameters are: {', '.join(paramMap.keys())}.")

    def applyFilter(self, filterOrder = 4, cutoffFrequency = 0.2):
        """
        Apply a Butterworth low-pass filter to the noisy signal.
        :param filterOrder: Order of the Butterworth filter (default is 4).
        :param cutoffFrequency: Cutoff frequency for the low-pass filter (default is 0.2).
        """
        if self.noisySignal is None:
            raise ValueError("Noisy signal is not generated. Please call 'generateNoisySignal' first.")

        # Design Butterworth low-pass filter
        [filterCoefficientsB, filterCoefficientsA] = signal.butter(filterOrder, cutoffFrequency, 'low', analog = False)

        self.filteredSignal = signal.filtfilt(filterCoefficientsB, filterCoefficientsA, self.noisySignal)

        print("Filter applied.")

    def computeFFT(self, signalToTransform):
        """
        Compute the Fast Fourier Transform (FFT) of a signal.
        :param signalToTransform: The signal to transform.
        :return: Frequencies and magnitudes of the Fourier Transform.
        """
        if signalToTransform is None:
            raise ValueError("Signal is None. Please provide a valid signal.")

        numSamples = len(signalToTransform)
        sampleSpacing = self.timeVector[1] - self.timeVector[0] \
            if len(self.timeVector) > 1 \
            else 1

        frequencies = np.fft.fftfreq(numSamples, sampleSpacing)
        magnitudes = np.abs(np.fft.fft(signalToTransform))

        return frequencies, magnitudes

    def setDampedSineWaveParameters(self, amplitudeParam, frequencyParam, phaseParam, decayRateParam):
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

    def setDampedSineWaveBounds(self, lower, upper):
        """
        Set the bounds for the damped sine wave fitting parameters.
        Default: ([0, 0, -np.pi, 0], [np.inf, np.inf, np.pi, np.inf])
        :param lower: Lower bounds for the parameters.
        :param upper: Upper bounds for the parameters.
        """
        if len(lower) != 4 or len(upper) != 4:
            raise ValueError("Bounds should be lists of length 4.")
        self.bounds = (lower, upper)

    def fitDampedSineWave(self):
        """
        Fit a damped sine wave to the filtered signal using nonlinear least squares.
        default sine wave parameters: amplitudeParam = 1.0, frequencyParam = 10.0, phaseParam = 0.0, decayRateParam = 0.1
        change with setDampedSineWaveParameters().
        """
        if self.filteredSignal is None:
            raise ValueError("Filtered signal is not available. Please apply the filter first 'applyFilter()'.")

        def dampedSineFunc(time, amplitude, frequency, phase, decayRate):
            return amplitude * np.exp(-decayRate * time) * np.sin(2 * np.pi * frequency * time + phase)

        def residualsDamped(params, time, data):
            return dampedSineFunc(time, *params) - data

        initialParams = np.array([self.amplitudeParam, self.frequencyParam, self.phaseParam, self.decayRateParam])

        try:
            result = optimize.least_squares(residualsDamped, initialParams, bounds = self.bounds, args = (self.timeVector, self.filteredSignal))
            self.optimalParamsDamped = result.x
        except Exception as e:
            raise RuntimeError(f"Optimization failed: {e}")

        self.fittedSignalDamped = dampedSineFunc(self.timeVector, *self.optimalParamsDamped)

        print("Damped sine wave fitted.")

    def performTTest(self):
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

    def plotResults(self):
        """
        Generate and save plots of the results including the signals and their Fourier Transforms.
        """
        if self.noisySignal is None or self.filteredSignal is None or self.fittedSignalDamped is None:
            raise ValueError("Noisy signal, filtered signal, and fitted signal must be available.")

        try:
            xfNoisy, yfNoisy = self.computeFFT(self.noisySignal)
            xfFiltered, yfFiltered = self.computeFFT(self.filteredSignal)

            plt.figure(figsize=(14, 10))

            # Time-domain plots
            plt.subplot(3, 2, 1)
            plt.plot(self.timeVector, self.noisySignal, label='Noisy Signal')
            plt.plot(self.timeVector, self.filteredSignal, label='Filtered Signal')
            plt.legend()
            plt.title('Time-Domain Signal Processing')

            # Fitting results
            plt.subplot(3, 2, 2)
            plt.plot(self.timeVector, self.filteredSignal, label='Filtered Signal')
            plt.plot(self.timeVector, self.fittedSignalDamped, label='Fitted Damped Sine Wave')
            plt.legend()
            plt.title('Fitting Damped Sine Wave')

            # Fourier Transform of noisy signal
            plt.subplot(3, 2, 3)
            plt.plot(xfNoisy, yfNoisy, label='FFT of Noisy Signal')
            plt.xlim(0, 50)  # Limit x-axis for better visibility
            plt.title('Fourier Transform of Noisy Signal')

            # Fourier Transform of filtered signal
            plt.subplot(3, 2, 4)
            plt.plot(xfFiltered, yfFiltered, label='FFT of Filtered Signal')
            plt.xlim(0, 50)  # Limit x-axis for better visibility
            plt.title('Fourier Transform of Filtered Signal')

            # Scatter plot for fitted vs filtered signal
            plt.subplot(3, 2, 5)
            plt.scatter(self.filteredSignal, self.fittedSignalDamped, alpha=0.5)
            plt.xlabel('Filtered Signal')
            plt.ylabel('Fitted Signal')
            plt.title(f'Scatter Plot\nT-statistic: {self.timeStatistic:.2f}, p-value: {self.pValue:.2e}')

            # Save and show the plot
            plt.tight_layout()
            saveDir = '../plots/'
            os.makedirs(saveDir, exist_ok = True)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            plt.savefig(f'{saveDir}plotResults_{timestamp}.png')
            plt.show()
            print("Results plotted and saved.")
        except Exception as e:
            raise RuntimeError(f"Plotting failed: {e}")

    def printResults(self):
        """
        Print the results of the fitting and statistical analysis.
        """
        if self.optimalParamsDamped is None or self.timeStatistic is None or self.pValue is None:
            raise ValueError("Optimal parameters, t-statistic, and p-value must be available. Please run 'fitDampedSineWave()' and 'performTTest()' first.")

        print(f"Optimal parameters for damped sine wave fitting: "
              f"A={self.optimalParamsDamped[0]:.2f}, "
              f"Frequency={self.optimalParamsDamped[1]:.2f}, "
              f"Phase={self.optimalParamsDamped[2]:.2f}, "
              f"Decay Rate={self.optimalParamsDamped[3]:.2f}")
        print(f"T-statistic: {self.timeStatistic:.2f}")
        print(f"P-value: {self.pValue:.2e}")
