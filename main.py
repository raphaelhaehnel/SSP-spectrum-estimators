import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import fft
from estimator import Estimator


def display_analytic_spectrum(omega: np.ndarray, Sxx1: np.ndarray, Sxx2: np.ndarray):
    """
    Shows the analytic spectrum of the signals x1 and x2
    """

    plt.figure()
    plt.title("Analytic computation of the spectrum")
    plt.plot(omega, Sxx1, label="$S_{XX_1}$", color="tab:blue")
    plt.plot(omega, Sxx2, label="$S_{XX_2}$", color="tab:orange")
    plt.legend()
    plt.grid()
    plt.xlabel(r"$\omega$")
    plt.ylabel(r"$S_{XX}$")


def display_samples(x1: np.ndarray, x2: np.ndarray):
    """
    Shows the samples of the signals x1 and x2
    """

    fig, ax = plt.subplots(2, sharex=True)
    fig.suptitle("Signals x1[n] and x2[n]")

    ax[0].plot(x1, color="tab:blue")
    ax[0].grid()
    ax[0].set_ylabel(r"$x_1[n]$")
    ax[0].set_xlabel(r"$n$")

    ax[1].plot(x2, color="tab:orange")
    ax[1].grid()
    ax[1].set_ylabel(r"$x_2[n]$")
    ax[1].set_xlabel(r"$n$")


def generate_x1_signal(L: int, sigma_1: float):
    """
    Generates the signal x1 by sampling random values for w1, and filtering to simulate an AM process
    """

    w_1 = sigma_1 * np.random.randn(L)
    x1 = signal.lfilter([1, -3, -4], [1], w_1)
    return x1


def generate_x2_signal(L: int, sigma_2: float):
    """
    Generates the signal x2 by sampling random values for w2, and filtering to simulate an AR process.
    Because the AR process depends on initial condition, we use only the second half of the signal
    """

    w_2 = sigma_2 * np.random.randn(2 * L)
    x2 = signal.lfilter([1], [1, -0.7], w_2)
    x2 = x2[L:]
    return x2


def generate_x2_initial(L: int):
    """ """
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
    x_periodogram = 1 / L * abs(fft.fft(x, M)) ** 2
    x_periodogram_half = x_periodogram[: int(M / 2) + 1]

    return x_periodogram_half / 2


def compute_correlogram(x: np.ndarray, L: int, M: int):
    Rx = 1 / L * np.correlate(x, x, mode="full")
    x_correlogram = np.abs(fft.fft(Rx, M))
    x_correlogram_half = x_correlogram[: int(M / 2) + 1]

    return x_correlogram_half / 2


def mean_periodogram_x(Mc: int, L: int, M: int, signal: str, sigma: float):
    mean_periodogram = np.zeros(2 * L + 1)

    for i in range(Mc):
        x = (
            generate_x1_signal(2 * L + 1, sigma)
            if signal == "x1"
            else generate_x2_signal(2 * L + 1, sigma)
        )
        x_periodogram = compute_periodogram(x, L, M)
        mean_periodogram += x_periodogram

    return mean_periodogram / Mc


def display_estimators(
    k_half: np.ndarray,
    x1_correlogram: np.ndarray,
    x1_periodogram: np.ndarray,
    x1_mean_period: np.ndarray,
    Sxx1: np.ndarray,
    x2_correlogram: np.ndarray,
    x2_periodogram: np.ndarray,
    x2_mean_period: np.ndarray,
    Sxx2: np.ndarray,
):
    fig, ax = plt.subplots(2, sharex=True)
    fig.suptitle("Estimators")
    ax[0].plot(k_half, x1_correlogram, color="tab:purple", label="Correlogram")
    ax[0].plot(k_half, x1_periodogram, color="tab:green", label="Periodogram")
    ax[0].plot(
        k_half, x1_mean_period, color="tab:blue", label="Monte-Carlo Periodogram"
    )
    ax[0].plot(k_half, Sxx1, color="tab:red", label="Analytic")
    ax[0].set_xlabel(r"$\omega$")
    ax[0].set_ylabel(r"$\hat{S}_{XX_1}$")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(k_half, x2_correlogram, color="tab:purple", label="Correlogram")
    ax[1].plot(k_half, x2_periodogram, color="tab:green", label="Periodogram")
    ax[1].plot(
        k_half, x2_mean_period, color="tab:blue", label="Monte-Carlo Periodogram"
    )
    ax[1].plot(k_half, Sxx2, color="tab:red", label="Analytic")
    ax[1].set_xlabel(r"$\omega$")
    ax[1].set_ylabel(r"$\hat{S}_{XX_2}$")
    ax[1].legend()
    ax[1].grid()


if __name__ == "__main__":
    # Question 1
    # For the first question, we want to display the result of the analytic computation of the spectrum
    # for x_1 and x_2

    # Resolution of the spectrum
    L_samples = 2049

    # Defining omega as x asis
    omega = np.linspace(0, np.pi, L_samples)

    Sxx1 = 1 + 9 / 13 * np.cos(omega) - 4 / 13 * np.cos(2 * omega)
    Sxx2 = 0.51 / (1.49 - 1.4 * np.cos(omega))

    display_analytic_spectrum(omega, Sxx1, Sxx2)

    # Question 2

    sigma_1 = np.sqrt(1 / 26)
    sigma_2 = np.sqrt(0.51)

    L = 1024

    x1 = generate_x1_signal(L, sigma_1)
    x2 = generate_x2_signal(L, sigma_2)

    # Signal that we are generating by defining x2[0] manually
    x2_initial = generate_x2_initial(L)

    display_samples(x1, x2)

    M = 4096
    k = np.linspace(0, 2 * np.pi, M)
    k_half = np.linspace(0, np.pi, int(M / 2) + 1)

    # Computation of the periodogram
    x1_periodogram = compute_periodogram(x1, L, M)
    x2_periodogram = compute_periodogram(x2, L, M)

    # Computation of the Correlogram
    x1_correlogram = compute_correlogram(x1, L, M)
    x2_correlogram = compute_correlogram(x2, L, M)

    # Monte-carlo simulation
    Mc = 100

    estimator_periodogram_x1 = Estimator(Mc, L, M, "x1", Sxx1, sigma_1)
    estimator_periodogram_x2 = Estimator(Mc, L, M, "x2", Sxx2, sigma_2)

    # Displays the 4 estimators in one window
    display_estimators(
        k_half,
        x1_correlogram,
        x1_periodogram,
        estimator_periodogram_x1.mean,
        Sxx1,
        x2_correlogram,
        x2_periodogram,
        estimator_periodogram_x2.mean,
        Sxx2,
    )

    plt.show()
