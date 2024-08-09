import numpy as np
from src.signal_processor import SignalProcessor

def run_example():
    # Generates a time vector for a duration of 2 seconds with a sampling rate of 1500 Hz. Consider importing or modifying your time vector
    duration = 2.0  # seconds
    sampling_rate = 1500  # Hz
    timeVector = np.linspace(0, duration, int(duration * sampling_rate), endpoint = False)

    # Initialize the SignalProcessor with time vector
    processor = SignalProcessor(timeVector)

    # Generate a noisy signal
    processor.generateNoisySignal()
    print("Noisy signal generated.")

    # Retrieve noisySignal
    noisySignal = processor.getParameter('noisySignal')
    print("Noisy Signal:", noisySignal)

    # Apply a low-pass filter to the noisy signal
    processor.applyFilter()
    print("Applied low-pass filter.")

    # Retrieve and print the filtered signal
    filteredSignal = processor.getParameter('filteredSignal')
    print("Filtered Signal:", filteredSignal)

    # Fit a damped sine wave to the filtered signal with default parameters
    processor.fitDampedSineWave()
    print("Fitted damped sine wave.")

    # Fit a damped sine wave to the filtered signal with custom parameters
    processor.setDampedSineWaveParameters(3.0, 12.0, np.pi / 6, 0.3)
    processor.setDampedSineWaveBounds([0, 0, -np.pi/2, 0], [10, 20, np.pi/2, 1])
    processor.fitDampedSineWave()
    print("Fitted damped sine wave.")

    # Retrieve and print the fitted damped sine wave signal
    fittedSignalDamped = processor.getParameter('fittedSignalDamped')
    print("Fitted Damped Sine Wave Signal:", fittedSignalDamped)

    # Perform a t-test between the filtered signal and the fitted damped sine wave
    processor.performTTest()
    print("Performed timeVector-test.")

    # Retrieve and print t-test results
    timeStatistic = processor.getParameter('timeStatistic')
    pValue = processor.getParameter('pValue')
    print("T-Statistic:", timeStatistic)
    print("P-Value:", pValue)

    # Plot and save results
    processor.plotResults()
    print("Results plotted and saved as 'plot_results.png'.")

    # Print the fitting results and statistical analysis
    processor.printResults()

    # Print the fitting results and statistical analysis
    fittingResults = processor.getFittingResults()
    print("Fitting Results:", fittingResults)

if __name__ == "__main__":
    run_example()
