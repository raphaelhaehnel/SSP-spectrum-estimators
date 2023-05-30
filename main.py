import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import fft
from estimator import Estimator


def display_analytic_spectrum(omega: np.ndarray, Sxx1: np.ndarray, Sxx2: np.ndarray):
    """
    Shows the analytic spectrum of the signals x1 and x2

    Parameters
    ----------
    omega : Array of frequencies
    Sxx1 : Array of the spectrum of signal x1
    Sxx2 : Array of the spectrum of signal x2
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

    Parameters
    ----------
    x1 : Array of the samples of the signal x1
    x2 : Array of the samples of the signal x2
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
    x_periodogram_half = x_periodogram[: int(M / 2) + 1] / 2

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

    Rx = 1 / L * np.correlate(x, x, mode="full")
    x_correlogram = np.abs(fft.fft(Rx, M))
    x_correlogram_half = x_correlogram[: int(M / 2) + 1] / 2

    return x_correlogram_half


def compute_bartlett(x: np.ndarray, M: int, K: int, L: int):
    """
    Implementation of the Bartlett estimator

    Parameters
    ----------
    x : Array of samples of a signal
    M : Length of the fourier transform (resolution of the frequency plane)
    K : Number of sections of the signal
    L : Length of one section from the signal

    Returns
    -------
    x_correlogram_half : Correlogram of the signal
    """

    x_splitted = np.split(x, K)
    x_barlet = np.zeros(M)
    for sub_x in x_splitted:
        x_barlet += 1 / L * np.abs(fft.fft(sub_x, M)) ** 2

    x_barlet = x_barlet[: int(M / 2) + 1] / K

    return x_barlet


def compute_welch(x: np.ndarray, M: int, K: int, L: int, D: int):
    """
    Implementation of the Welch estimator
    """

    x_splitted = [x[i * D : i * D + L] for i in range(K)]

    x_welch = np.zeros(M)

    for sub_x in x_splitted:
        x_welch += 1 / L * np.abs(fft.fft(sub_x, M)) ** 2

    x_welch = x_welch[: int(M / 2) + 1]

    return x_welch / len(x_splitted)


def compute_BT(x: np.ndarray, L: int, M: int, L_BT: int):
    """
    Implementation of the Blackman Tukey estimator
    """

    Rx = 1 / L * np.correlate(x, x, mode="full")
    Rx_windowed = Rx[L - 1 - L_BT : L + L_BT]
    x_BT = np.abs(fft.fft(Rx_windowed, M))
    x_BT_half = x_BT[: int(M / 2) + 1]  # TODO why here I don't need to divide by 2 ?

    return x_BT_half


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
    x1_bartlett16: np.ndarray,
    x1_bartlett64: np.ndarray,
    x1_welch_61: np.ndarray,
    x1_bt_4: np.ndarray,
    x1_bt_2: np.ndarray,
    Sxx1: np.ndarray,
    x2_correlogram: np.ndarray,
    x2_periodogram: np.ndarray,
    x2_mean_period: np.ndarray,
    x2_bartlett16: np.ndarray,
    x2_bartlett64: np.ndarray,
    x2_welch_61: np.ndarray,
    x2_bt_4: np.ndarray,
    x2_bt_2: np.ndarray,
    Sxx2: np.ndarray,
):
    fig, ax = plt.subplots(2, sharex=True)
    fig.suptitle("Estimators")
    # ax[0].plot(k_half, x1_correlogram, color="tab:purple", label="Correlogram")
    # ax[0].plot(k_half, x1_periodogram, color="tab:green", label="Periodogram")
    # ax[0].plot(
    #     k_half, x1_mean_period, color="tab:blue", label="Monte-Carlo Periodogram"
    # )
    ax[0].plot(k_half, x1_bartlett16, color="tab:olive", label="Bartlet16")
    # ax[0].plot(k_half, x1_bartlett64, color="tab:pink", label="Bartlet64")
    ax[0].plot(k_half, x1_welch_61, color="tab:pink", label="Welch61")
    ax[0].plot(k_half, x1_bt_4, color="tab:blue", label="Blackman Tukey4")
    ax[0].plot(k_half, x1_bt_2, color="tab:purple", label="Blackman Tukey2")
    ax[0].plot(k_half, x1_welch_61, color="tab:pink", label="Welch61")
    ax[0].plot(k_half, Sxx1, color="tab:red", label="Analytic")
    ax[0].set_xlabel(r"$\omega$")
    ax[0].set_ylabel(r"$\hat{S}_{XX_1}$")
    ax[0].legend()
    ax[0].grid()

    # ax[1].plot(k_half, x2_correlogram, color="tab:purple", label="Correlogram")
    # ax[1].plot(k_half, x2_periodogram, color="tab:green", label="Periodogram")
    # ax[1].plot(
    #     k_half, x2_mean_period, color="tab:blue", label="Monte-Carlo Periodogram"
    # )
    ax[1].plot(k_half, x2_bartlett16, color="tab:olive", label="Bartlet16")
    # ax[1].plot(k_half, x2_bartlett64, color="tab:pink", label="Bartlet64")
    ax[1].plot(k_half, x2_welch_61, color="tab:pink", label="Welch61")
    ax[1].plot(k_half, x2_bt_4, color="tab:blue", label="Blackman Tukey4")
    ax[1].plot(k_half, x2_bt_2, color="tab:purple", label="Blackman Tukey2")
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
    x2_initial = generate_x2_initial(L, sigma_2)

    display_samples(x1, x2)

    M = 4096
    k = np.linspace(0, 2 * np.pi, M)
    k_half = np.linspace(0, np.pi, int(M / 2) + 1)

    # Computation of the Periodogram
    x1_periodogram = compute_periodogram(x1, L, M)
    x2_periodogram = compute_periodogram(x2, L, M)

    # Computation of the Correlogram
    x1_correlogram = compute_correlogram(x1, L, M)
    x2_correlogram = compute_correlogram(x2, L, M)

    x1_bartlett_16 = compute_bartlett(x1, M, K=16, L=64)
    x2_bartlett_16 = compute_bartlett(x2, M, K=16, L=64)

    x1_bartlett_64 = compute_bartlett(x1, M, K=64, L=16)
    x2_bartlett_64 = compute_bartlett(x2, M, K=64, L=16)

    x1_welch_61 = compute_welch(x1, M, K=61, L=64, D=64 - 48)
    x2_welch_61 = compute_welch(x2, M, K=61, L=64, D=64 - 48)

    x1_welch_253 = compute_welch(x1, M, K=253, L=16, D=16 - 12)
    x2_welch_253 = compute_welch(x2, M, K=253, L=16, D=16 - 12)

    x1_bt_4 = compute_BT(x1, L, M, L_BT=4)
    x1_bt_2 = compute_BT(x1, L, M, L_BT=2)

    x2_bt_4 = compute_BT(x2, L, M, L_BT=4)
    x2_bt_2 = compute_BT(x2, L, M, L_BT=2)

    # Monte-carlo simulation
    Mc = 100

    estimator_periodogram_x1 = Estimator(Mc, L, M, "x1", Sxx1, sigma_1)
    estimator_periodogram_x2 = Estimator(Mc, L, M, "x2", Sxx2, sigma_2)

    # Displays the estimators in one window
    display_estimators(
        k_half,
        x1_correlogram,
        x1_periodogram,
        estimator_periodogram_x1.mean,
        x1_bartlett_16,
        x1_bartlett_64,
        x1_welch_61,
        x1_bt_4,
        x1_bt_2,
        Sxx1,
        x2_correlogram,
        x2_periodogram,
        estimator_periodogram_x2.mean,
        x2_bartlett_16,
        x2_bartlett_64,
        x2_welch_61,
        x2_bt_4,
        x2_bt_2,
        Sxx2,
    )

    plt.show()
