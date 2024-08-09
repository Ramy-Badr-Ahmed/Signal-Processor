from signal_processor import SignalProcessor
import numpy as np

def runProcessing():
    """
    Run the signal processing steps using the default parameters of the SignalProcessor class.
    """
    timeVector = np.linspace(0, 1, 1000, endpoint = False)   # Consider importing or modifying your time vector
    processor = SignalProcessor(timeVector)
    processor.generateNoisySignal()
    processor.applyFilter()
    processor.fitDampedSineWave()
    processor.performTTest()
    processor.plotResults()
    processor.plotInteractiveResults()
    processor.printResults()

if __name__ == "__main__":
    try:
        runProcessing()
    except Exception as e:
        print(f"An error occurred: {e}")