from enum import Enum
import numpy as np
from scipy import fft
from scipy import signal

Signal = Enum("Signal", ["x1", "x2"])
Estimate = Enum("Estimate", ["PERIODOGRAM", "BARTLETT", "WELCH", "BLACKMAN_TUKEY"])


def generate_analytic_spectrum(x_str: Signal, omega: np.ndarray):
    """
    Generate analytic spectrum of a signal

    Parameters
    ----------
    x_str : Name of the signal we want to generate
    omega : Array of frequencies (x-axis)

    Returns
    -------
    Sxx: The analytic spectrum of the desired signal
    """

    if x_str == Signal.x1:
        return 1 + 9 / 13 * np.cos(omega) - 4 / 13 * np.cos(2 * omega)
    if x_str == Signal.x2:
        return 0.51 / (1.49 - 1.4 * np.cos(omega))

    error = f"Invalid parameter '{x_str}'"
    raise ValueError(error)


def generate_x1_signal(L: int, sigma_1: float):
    """
    Generates the signal x1 by sampling random values for w1, and filtering to simulate an AM process

    Parameters
    ----------
    L : Lenth of the signal x1
    sigma_1 : Standard deviation of the noise w1

    Returns
    -------
    x1 : Array of samples of the signal x1
    """

    w_1 = sigma_1 * np.random.randn(L)
    x1 = signal.lfilter([1, -3, -4], [1], w_1)
    return x1


def generate_x2_signal(L: int, sigma_2: float):
    """
    Generates the signal x2 by sampling random values for w2, and filtering to simulate an AR process.
    Because the AR process depends on initial condition, we use only the second half of the signal

    Parameters
    ----------
    L : Lenth of the signal x2
    sigma_2 : Standard deviation of the noise w2

    Returns
    -------
    x2 : Array of samples of the signal x2
    """

    w_2 = sigma_2 * np.random.randn(2 * L)
    x2 = signal.lfilter([1], [1, -0.7], w_2)
    x2 = x2[L:]
    return x2


def generate_x2_initial(L: int, sigma_2: float):
    """
    Generates the signal x2 by sampling random values for w2, and filtering to simulate an AR process.
    Because the AR process depends on initial condition, we use only the second half of the signal.
    This function is generating the signal by sampling x2[0] according to normal distribution

    Parameters
    ----------
    L : Lenth of the signal x2
    sigma_2 : Standard deviation of the noise w2

    Returns
    -------
    x2 : Array of samples of the signal x2
    """

    w_2 = sigma_2 * np.random.randn(2 * L)
    x2 = np.zeros(L)
    initial = np.random.randn()
    for i in range(L):
        if i == 0:
            x2[i] = 0.7 * initial + w_2[i]
        else:
            x2[i] = 0.7 * x2[i - 1] + w_2[i]
    return x2


def compute_periodogram(x: np.ndarray, L: int, M: int):
    """
    Implementation of the Periodogram estimator

    Parameters
    ----------
    x : Array of samples of a signal
    L : Length of the signal
    M : Length of the fourier transform (resolution of the frequency plane)

    Returns
    -------
    x_periodogram_half : Periodogram of the signal
    """

    x_periodogram = 1 / L * abs(fft.fft(x, M)) ** 2

    # The fourier transform is duplicated, so we are taking only the first half of the transform
    x_periodogram_half = x_periodogram[: int(M / 2) + 1]

    return x_periodogram_half


def compute_correlogram(x: np.ndarray, L: int, M: int):
    """
    Implementation of the Correlogram estimator

    Parameters
    ----------
    x : Array of samples of a signal
    L : Length of the signal
    M : Length of the fourier transform (resolution of the frequency plane)

    Returns
    -------
    x_correlogram_half : Correlogram of the signal
    """

    # Compute the autocorrelation (biased)
    Rx = 1 / L * np.correlate(x, x, mode="full")

    # Compute the fourier transform
    x_correlogram = np.abs(fft.fft(Rx, M))

    x_correlogram_half = x_correlogram[: int(M / 2) + 1]

    return x_correlogram_half


def compute_bartlett(x: np.ndarray, M: int, K: int, L_section: int):
    """
    Implementation of the Bartlett estimator

    Parameters
    ----------
    x : Array of samples of a signal
    M : Length of the fourier transform (resolution of the frequency plane)
    K : Number of sections of the signal
    L_section : Length of one section from the signal

    Returns
    -------
    x_barlet_half : Bartlett estimate of the signal
    """

    x_splitted = np.split(x, K)
    x_barlet = np.zeros(M)
    for sub_x in x_splitted:
        x_barlet += 1 / L_section * np.abs(fft.fft(sub_x, M)) ** 2

    x_barlet_half = x_barlet[: int(M / 2) + 1] / K

    return x_barlet_half


def compute_welch(x: np.ndarray, M: int, K: int, L_section: int, D: int):
    """
    Implementation of the Welch estimator

    Parameters
    ----------
    x : Array of samples of a signal
    M : Length of the fourier transform (resolution of the frequency plane)
    K : Number of sections of the signal
    L_section : Length of one section from the signal
    D : Offset distance (D is equal to L minus the overlapping)

    Returns
    -------
    x_welch_half : Welch estimate of the signal
    """

    x_splitted = [x[i * D : i * D + L_section] for i in range(K)]

    x_welch = np.zeros(M)

    for sub_x in x_splitted:
        x_welch += 1 / L_section * np.abs(fft.fft(sub_x, M)) ** 2

    x_welch_half = x_welch[: int(M / 2) + 1] / len(x_splitted)

    return x_welch_half


def compute_BT(x: np.ndarray, L: int, M: int, L_BT: int):
    """
    Implementation of the Blackman Tukey estimator

    Parameters
    ----------
    x : Array of samples of a signal
    L : Length of the whole signal
    M : Length of the fourier transform (resolution of the frequency plane)
    L_BT : size of the window of the blackman-tukey : -L_BT < l < L_BT

    Returns
    -------
    x_BT_half : Blackman Tukey estimate of the signal
    """

    Rx = 1 / L * np.correlate(x, x, mode="full")
    Rx_windowed = Rx[L - 1 - L_BT : L + L_BT]
    x_BT = np.abs(fft.fft(Rx_windowed, M))
    x_BT_half = x_BT[: int(M / 2) + 1]  # TODO why here I don't need to divide by 2 ?

    return x_BT_half
