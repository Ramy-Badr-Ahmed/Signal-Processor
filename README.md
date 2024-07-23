![Python](https://img.shields.io/badge/Python-3670A0?style=plastic&logo=python&logoColor=ffdd54)  

![GitHub](https://img.shields.io/github/license/Ramy-Badr-Ahmed/SignalProcessor)

# SignalProcessor

This repository provides a Python package for generating, filtering, fitting, and analyzing signals. The package includes functionalities for creating noisy signals, applying filters, fitting damped sine waves, and performing statistical analysis.

### Overview

- Generate noisy sine wave signals
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
An example demonstrating generating a signal, applying filters, fitting models, and performing analysis, exists under `examples` directory (refer to `run_example.py`)

>[!Note]
> An example plot has been uploaded to the `plots` directory.

### Example Usage

Generate a Noisy Signal

```shell
import numpy as np
from src.signal_processor import SignalProcessor

timeVector = np.linspace(0, 1, 1000, endpoint = False)  # Or consider importing or modifying your time vector

processor = SignalProcessor(timeVector)
   
processor.generateNoisySignal(frequency = 20, noiseStdDev = 0.6)  # or empty: default frequency = 10, noiseStdDev = 0.5
```

Apply Butterworth low-pass filter

```shell
processor.applyFilter(filterOrder = 2, cutoffFrequency = 0.8)   # or empty: default filterOrder = 4, cutoffFrequency = 0.2
```

Fit a damped sine wave to the filtered signal

```shell
processor.fitDampedSineWave()   # default sine wave parameters: amplitudeParam = 1.0, frequencyParam = 10.0, phaseParam = 0.0, decayRateParam = 0.1
```

Perform a t-test between the filtered signal and the fitted damped sine wave

```shell
processor.performTTest()
```

Plot and save the results (will be saved under `plots` directory)

```shell
processor.plotResults()
```

Print the fitting and statistical results
```shell
processor.printResults()
```
