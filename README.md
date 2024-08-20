![Python](https://img.shields.io/badge/Python-3670A0?style=plastic&logo=python&logoColor=ffdd54) ![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=plastic&logo=scipy&logoColor=%white) ![NumPy](https://img.shields.io/badge/Numpy-777BB4.svg?style=plastic&logo=numpy&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-239120.svg?style=plastic&logo=plotly&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-%233F4F75.svg?style=plastic&logo=plotly&logoColor=white)  ![GitHub](https://img.shields.io/github/license/Ramy-Badr-Ahmed/SignalProcessor?style=plastic)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13286179.svg)](https://doi.org/10.5281/zenodo.13286179)

# SignalProcessor

This repository provides a Python package for generating, filtering, fitting, and analyzing signals. The package includes functionalities for creating noisy signals, applying filters, fitting damped sine waves, and performing statistical analysis.

### Overview

- Generate noisy sine wave signals (or import custom signals)
- Apply Butterworth low-pass filters
- Fit damped sine waves to filtered signals
- Perform t-tests between filtered signals and fitted models
- Compute and visualize Fourier Transforms

### Installation

1) Create and source virtual environment:
```shell
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
```
2) Install the dependencies:
```shell
pip install -r requirements.txt
```

### Running Tests
Using unittest

```shell
python -m unittest discover -s tests
```

### Example
An example demonstrating generating a signal, applying filters, fitting models, and performing analysis, exists in the `main.py`.

>[!Note]
> An example plot has been uploaded to the `plots` directory.

### Example Usage

Generate a Noisy Signal

```shell
import numpy as np
from src.signal_processor import SignalProcessor

timeVector = np.linspace(0, 1, 1000, endpoint = False)  # Or consider importing or modifying your time vector

generator = SignalGenerator(timeVector)
   
generator.generateNoisySignal(frequency = 20, noiseStdDev = 0.6)

  # or with defaults:
    processor.generateNoisySignal()   # frequency = 10, noiseStdDev = 0.5
```

Apply a Filter (`butter`, `bessel`, `highpass`). Default is `butter`.

```shell
from src.signal_filter import SignalFilter

filteredInstance = generator.generateNoisySignal() \
                            .applyFilter(filterType = 'butter', 
                                         filterOrder = 4, 
                                         cutOffFrequency = 0.2, 
                                         bType = 'lowpass')
    # Or with different filter parameters:
      filteredInstance.setFilterParameters('bessel', 5, 0.5, 'highpass').applyFilter()    
```

Fit a damped sine wave to the filtered signal

```shell    
from src.signal_fitter import SignalFitter

    # default sine wave parameters: amplitudeParam = 1.0, frequencyParam = 10.0, phaseParam = 0.0, decayRateParam = 0.1
fittedInstance = filteredInstance.fitDampedSineWave()

    # Or with custom parameters:
      fittedInstance.setDampedSineWaveParameters(3.0, 12.0, np.pi / 6, 0.3)
      fittedInstance.setDampedSineWaveBounds([0, 0, -np.pi/2, 0], [10, 20, np.pi/2, 1])
      fittedInstance.fitDampedSineWave()      
```

Perform a t-test between the filtered signal and the fitted damped sine wave

```shell
from src.statistical_analyzer import StatisticalAnalyzer

analyzedInstance = fittedInstance.analyzeFit()
tTestResults = analyzedInstance.getTTestResults()
print(f"T-test result: statistic={tTestResults[0]}, p-value={tTestResults[1]}")
```

Plot and save the results (will be saved under `plots` directory)

```shell
from src.signal_visualizer import SignalVisualizer

visualizer = SignalVisualizer(timeVector, generator.getNoisySignal(), 
                              filteredInstance.getFilteredSignal(), 
                              fittedInstance.getFittedSignal()
                              )
visualizer.plotResults()
visualizer.plotInteractiveResults()
```
