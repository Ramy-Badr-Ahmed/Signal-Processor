from src.statistical_analyzer import StatisticalAnalyzer
import matplotlib.pyplot as plt
import plotly.offline as pyo
import plotly.graph_objects as go
import numpy as np
import datetime
import os


class SignalVisualizer:
    def __init__(self, timeVector: np.ndarray, noisySignal: np.ndarray, filteredSignal: np.ndarray, fittedSignalDamped: np.ndarray):
        """
        Initialize the SignalVisualizer with signals to visualize.
        :param timeVector: The time vector.
        :param noisySignal: The original noisy signal.
        :param filteredSignal: The filtered signal.
        :param fittedSignalDamped: The fitted signal.
        """
        self.timeVector = timeVector
        self.noisySignal = noisySignal
        self.filteredSignal = filteredSignal
        self.fittedSignalDamped = fittedSignalDamped

        self.timeStatistic, self.pValue = StatisticalAnalyzer(self.filteredSignal, self.fittedSignalDamped).getTTestResults()

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

    def plotInteractiveResults(self) -> None:
        """
        Generate interactive plots of the results including the signals and their Fourier Transforms.
        Each plot is saved in a separate HTML file.
        """
        if self.noisySignal is None or self.filteredSignal is None or self.fittedSignalDamped is None:
            raise ValueError("Noisy signal, filtered signal, and fitted signal must be available.")
        try:
            saveDir = '../plots/'
            os.makedirs(saveDir, exist_ok = True)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            xfNoisy, yfNoisy = self.computeFFT(self.noisySignal)
            xfFiltered, yfFiltered = self.computeFFT(self.filteredSignal)

            # Time-domain plots for noisy and filtered signals
            fig_time_domain = go.Figure()
            fig_time_domain.add_trace(go.Scatter(x = self.timeVector, y = self.noisySignal, mode = 'lines', name = 'Noisy Signal'))
            fig_time_domain.add_trace(go.Scatter(x = self.timeVector, y = self.filteredSignal, mode = 'lines', name = 'Filtered Signal'))
            fig_time_domain.update_layout(title='Time-Domain Signal Processing',
                                          xaxis_title = 'Time (s)',
                                          yaxis_title = 'Amplitude',
                                          margin = dict(l = 0, r = 0, t = 50, b = 0)
                                          )

            pyo.plot(fig_time_domain, filename = f'{saveDir}interactive_TimeDomain_{timestamp}.html', auto_open = True)

            # Fitting results
            fig_fitting = go.Figure()
            fig_fitting.add_trace(go.Scatter(x = self.timeVector, y = self.filteredSignal, mode = 'lines', name = 'Filtered Signal'))
            fig_fitting.add_trace(go.Scatter(x = self.timeVector, y = self.fittedSignalDamped, mode = 'lines', name = 'Fitted Damped Sine Wave'))
            fig_fitting.update_layout(title = 'Fitting Damped Sine Wave',
                                      xaxis_title = 'Time (s)',
                                      yaxis_title = 'Amplitude',
                                      margin = dict(l = 0, r = 0, t = 50, b = 0)
                                      )

            pyo.plot(fig_fitting, filename = f'{saveDir}interactive_Fitting_{timestamp}.html', auto_open = True)

            # Fourier Transform of noisy signal
            fig_fft_noisy = go.Figure()
            fig_fft_noisy.add_trace(go.Scatter(x = xfNoisy, y = yfNoisy, mode = 'lines', name = 'FFT of Noisy Signal'))
            fig_fft_noisy.update_layout(title='Fourier Transform of Noisy Signal',
                                        xaxis_title = 'Frequency (Hz)',
                                        yaxis_title = 'Amplitude',
                                        margin = dict(l = 0, r = 0, t = 50, b = 0)
                                        )

            pyo.plot(fig_fft_noisy, filename = f'{saveDir}interactive_FFT_Noisy_{timestamp}.html', auto_open = True)

            # Fourier Transform of filtered signal
            fig_fft_filtered = go.Figure()
            fig_fft_filtered.add_trace(go.Scatter(x = xfFiltered, y = yfFiltered, mode = 'lines', name = 'FFT of Filtered Signal'))
            fig_fft_filtered.update_layout(title='Fourier Transform of Filtered Signal',
                                           xaxis_title = 'Frequency (Hz)',
                                           yaxis_title = 'Amplitude',
                                           margin = dict(l = 0, r = 0, t = 50, b = 0)
                                           )

            pyo.plot(fig_fft_filtered, filename = f'{saveDir}interactive_FFT_Filtered_{timestamp}.html', auto_open = True)

            # Scatter plot for fitted vs filtered signal
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(x = self.filteredSignal, y = self.fittedSignalDamped, mode = 'markers', name = 'Fitted vs Filtered'))
            fig_scatter.update_layout(title = f'Scatter Plot\nT-statistic: {self.timeStatistic:.2f}, p-value: {self.pValue:.2e}',
                                      xaxis_title = 'Filtered Signal',
                                      yaxis_title = 'Fitted Signal',
                                      margin = dict(l = 0, r = 0, t = 50, b = 0)
                                      )
            fig_scatter.write_html(f'{saveDir}interactive_Scatter_{timestamp}.html')

            pyo.plot(fig_scatter, filename = f'{saveDir}interactive_Scatter_{timestamp}.html', auto_open = True)

            print("Interactive plots generated and saved.")
        except Exception as e:
            raise RuntimeError(f"Interactive plotting failed: {e}")