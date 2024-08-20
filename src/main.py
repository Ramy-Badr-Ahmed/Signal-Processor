from src.signal_generator import SignalGenerator
from src.signal_visualizer import SignalVisualizer
import numpy as np

def runProcessing():
    """
    Run the signal processing steps using the default parameters of the SignalProcessor classes.
    """
    timeVector = np.linspace(0, 1, 1000, endpoint = False)   # Consider importing or modifying your time vector

    noisyInstance = SignalGenerator(timeVector)

    filteredInstance = noisyInstance.generateNoisySignal().applyFilter()
    fittedInstance = filteredInstance.fitDampedSineWave()
    analyzedInstance = fittedInstance.analyzeFit()

    # Retrieve results
    noisySignal = noisyInstance.getNoisySignal()
    filteredSignal = filteredInstance.getFilteredSignal()
    fittedSignal = fittedInstance.getFittedSignal()
    tTestResults = analyzedInstance.getTTestResults()

    print(f"T-test result: statistic={tTestResults[0]}, p-value={tTestResults[1]}")

    visualizer = SignalVisualizer(timeVector, noisySignal, filteredSignal, fittedSignal)

    visualizer.plotResults()
    visualizer.plotInteractiveResults()

if __name__ == "__main__":
    try:
        runProcessing()
    except Exception as e:
        print(f"An error occurred: {e}")