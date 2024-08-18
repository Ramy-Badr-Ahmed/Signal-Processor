import matplotlib.pyplot as plt
import numpy as np
import os
import datetime

class SignalVisualizer:
    def __init__(self, time_vector: np.ndarray, noisySignal: np.ndarray, filteredSignal: np.ndarray, fittedSignal: np.ndarray):
        self.timeVector = time_vector
        self.noisySignal = noisySignal
        self.filteredSignal = filteredSignal
        self.fittedSignalDamped = fittedSignal

    def computeFFT(self, signalToTransform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

    def plotResults(self)-> None:
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
