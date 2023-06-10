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

### Output
