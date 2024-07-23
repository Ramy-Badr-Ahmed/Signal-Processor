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
    parameter = processor.getParameter('noisySignal')
    print(parameter)

    # Apply a low-pass filter to the noisy signal
    processor.applyFilter()
    print("Applied low-pass filter.")

    # Retrieve filteredSignal
    parameter = processor.getParameter('filteredSignal')
    print(parameter)

    # Fit a damped sine wave to the filtered signal
    processor.fitDampedSineWave()
    print("Fitted damped sine wave.")

    # Retrieve fittedSignalDamped
    parameter = processor.getParameter('fittedSignalDamped')
    print(parameter)

    # Perform a timeVector-test between the filtered signal and the fitted signal
    processor.performTTest()
    print("Performed timeVector-test.")

    # Retrieve timeStatistic
    timeStatistic = processor.getParameter('timeStatistic')
    pValue = processor.getParameter('pValue')
    print(timeStatistic, pValue)

    # Plot and save results
    processor.plotResults()
    print("Results plotted and saved as 'advanced_results.png'.")

    # Print the fitting results and statistical analysis
    processor.printResults()

if __name__ == "__main__":
    run_example()
