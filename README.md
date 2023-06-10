## PROJECT IN STATISTICAL SIGNAL PROCESSING (SSP)

This project has been done during the course "SSP" with Prof. Sharon Gannot, Bar-Ilan University,
2023, second semester.

The aim of this project is to implement different estimators to estimate spectrums and compare their efficiency.

The exercice deals with spectrum estimation of two signals: MA process and AR process. We're dealing with 5 estimators:

1. Correlogram
2. Periodogram
3. Barlett
4. Welch
5. Blackman Tukey

#### main.py

Run this script to display the different graphs

#### signal_processing.py

Contains all the functions to generate the signals and compute the spectrums

#### estimator.py

Defines the class Estimator. It automatically computes the different metrics for a given signal and a given estimator.

### Input

We are giving two signals as input:

1. MA process: $`x_1 [n]=w_1 [n]-3w_1 [n-1]-4w_1 [n-2]`$
2. AR process: $`x_2 [n]=0.7x_2 [n-1]+w_2 [n]`$

<img src="https://github.com/raphaelhaehnel/SSP-spectrum-estimators/assets/69756617/301e5d77-6c5f-40c3-b9c6-7a2482505c53" width="65%">

### Output

First we are computing analytically the spectrum, and we are displaying the results.
<img src="https://github.com/raphaelhaehnel/SSP-spectrum-estimators/assets/69756617/3f5cf822-3d43-44bc-8784-06af2f88c0de" width="65%">

We are computing the Correlogram and the Periodogram. We can see on the graph that they are the same.
<img src="https://github.com/raphaelhaehnel/SSP-spectrum-estimators/assets/69756617/561d56d9-1d3c-48ba-83cd-51e3d28dee98" width="65%">

Next we’ll show all the spectrums estimations before and after the Monte-Carlo simulation. Before:
<img src="https://github.com/raphaelhaehnel/SSP-spectrum-estimators/assets/69756617/01eb0ae2-68cb-4b5d-9cbf-6bfdfe343449" width="65%">

After:
<img src="https://github.com/raphaelhaehnel/SSP-spectrum-estimators/assets/69756617/9bffd1e8-f30d-4ba6-be28-7b15b057cb8e" width="65%">

And now we’ll display the different metrics: bias, variance and mean square error (variance and mse are shown on logarithm scale).

<img src="https://github.com/raphaelhaehnel/SSP-spectrum-estimators/assets/69756617/b60717ff-cc33-41e3-87a6-979d7d77b95c" width="65%">

<img src="https://github.com/raphaelhaehnel/SSP-spectrum-estimators/assets/69756617/f6615449-5fff-4c1d-bb82-544a67ad74dc" width="65%">

<img src="https://github.com/raphaelhaehnel/SSP-spectrum-estimators/assets/69756617/ad098078-ea86-434e-848a-2b3fb79cf61b" width="65%">


