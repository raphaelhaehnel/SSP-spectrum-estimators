import numpy as np
import matplotlib.pyplot as plt
from estimator import Estimate, Estimator
from signal_processing import *

# Standard deviation of the random process w1
SIGMA_1: float = np.sqrt(1 / 26)

# Standard deviation of the random process w2
SIGMA_2: float = np.sqrt(0.51)

# Length of the fourier transform (resolution of the frequency plane)
M = 4096


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
    """
    This function displays all the estimators in one graph

    Parameters
    ----------
    Takes all the estimators as parameters

    """
    fig, ax = plt.subplots(2, sharex=True)
    fig.suptitle("Estimators")
    # ax[0].plot(k_half, x1_correlogram, color="tab:purple", label="Correlogram")
    # ax[0].plot(k_half, x1_periodogram, color="tab:green", label="Periodogram")
    ax[0].plot(k_half, x1_mean_period, color="tab:blue", label="Monte-Carlo Periodogram")
    # ax[0].plot(k_half, x1_bartlett16, color="tab:olive", label="Bartlet16")
    # ax[0].plot(k_half, x1_bartlett64, color="tab:pink", label="Bartlet64")
    # ax[0].plot(k_half, x1_welch_61, color="tab:pink", label="Welch61")
    # ax[0].plot(k_half, x1_bt_4, color="tab:blue", label="Blackman Tukey4")
    # ax[0].plot(k_half, x1_bt_2, color="tab:purple", label="Blackman Tukey2")
    ax[0].plot(k_half, Sxx1, color="tab:red", label="Analytic")
    ax[0].set_xlabel(r"$\omega$")
    ax[0].set_ylabel(r"$\hat{S}_{XX_1}$")
    ax[0].legend()
    ax[0].grid()

    # ax[1].plot(k_half, x2_correlogram, color="tab:purple", label="Correlogram")
    # ax[1].plot(k_half, x2_periodogram, color="tab:green", label="Periodogram")
    ax[1].plot(k_half, x2_mean_period, color="tab:blue", label="Monte-Carlo Periodogram")
    # ax[1].plot(k_half, x2_bartlett16, color="tab:olive", label="Bartlet16")
    # ax[1].plot(k_half, x2_bartlett64, color="tab:pink", label="Bartlet64")
    # ax[1].plot(k_half, x2_welch_61, color="tab:pink", label="Welch61")
    # ax[1].plot(k_half, x2_bt_4, color="tab:blue", label="Blackman Tukey4")
    # ax[1].plot(k_half, x2_bt_2, color="tab:purple", label="Blackman Tukey2")
    ax[1].plot(k_half, Sxx2, color="tab:red", label="Analytic")
    ax[1].set_xlabel(r"$\omega$")
    ax[1].set_ylabel(r"$\hat{S}_{XX_2}$")
    ax[1].legend()
    ax[1].grid()


if __name__ == "__main__":
    # Question 1 #

    # For the first question, we want to display the result of the analytic computation of the spectrum
    # for x_1 and x_2

    # Resolution of the spectrum
    L_samples = 2049

    # Defining omega as x asis
    omega = np.linspace(0, np.pi, L_samples)

    # Generate the analytic spectrum of x1
    Sxx1 = generate_analytic_spectrum(Signal.x1, omega)

    # Generate the analytic spectrum of x2
    Sxx2 = generate_analytic_spectrum(Signal.x2, omega)

    # Display the analytic spectrum on a single graph
    display_analytic_spectrum(omega, Sxx1, Sxx2)

    # Question 2 #

    L = 1024

    x1 = generate_x1_signal(L, SIGMA_1)
    x2 = generate_x2_signal(L, SIGMA_2)

    # Signal that we are generating by defining x2[0] manually
    x2_initial = generate_x2_initial(L, SIGMA_2)

    display_samples(x1, x2)

    k = np.linspace(0, 2 * np.pi, M)
    k_half = np.linspace(0, np.pi, int(M / 2) + 1)

    # Computation of the Periodogram
    x1_periodogram = compute_periodogram(x1, L, M)
    x2_periodogram = compute_periodogram(x2, L, M)

    # Computation of the Correlogram
    x1_correlogram = compute_correlogram(x1, L, M)
    x2_correlogram = compute_correlogram(x2, L, M)

    # Computation of the Bartlett estimation with 16 sections
    x1_bartlett_16 = compute_bartlett(x1, M, K=16, L_section=64)
    x2_bartlett_16 = compute_bartlett(x2, M, K=16, L_section=64)

    # Computation of the Bartlett estimation with 64 sections
    x1_bartlett_64 = compute_bartlett(x1, M, K=64, L_section=16)
    x2_bartlett_64 = compute_bartlett(x2, M, K=64, L_section=16)

    # Computation of the Welch estimation with overlapping of 64
    x1_welch_61 = compute_welch(x1, M, K=61, L_section=64, D=64 - 48)
    x2_welch_61 = compute_welch(x2, M, K=61, L_section=64, D=64 - 48)

    # Computation of the Welch estimation with overlapping of 16
    x1_welch_253 = compute_welch(x1, M, K=253, L_section=16, D=16 - 12)
    x2_welch_253 = compute_welch(x2, M, K=253, L_section=16, D=16 - 12)

    # Computation of the Blackman Tukey estimation with a window of 4
    x1_bt_4 = compute_BT(x1, L, M, L_BT=4)
    x2_bt_4 = compute_BT(x2, L, M, L_BT=4)

    # Computation of the Blackman Tukey estimation with a window of 2
    x1_bt_2 = compute_BT(x1, L, M, L_BT=2)
    x2_bt_2 = compute_BT(x2, L, M, L_BT=2)

    # The number of simulation (Monte-carlo)
    Mc = 100

    # my_estimator = Estimator(
    #     Estimate.PERIODOGRAM,
    #     Signal.x1,
    #     Mc,
    # )

    # estimator_periodogram_x1 = Estimator(Mc, L, M, "x1", Sxx1, SIGMA_1)
    # estimator_periodogram_x2 = Estimator(Mc, L, M, "x2", Sxx2, SIGMA_2)

    estimator_periodogram_x1 = Estimator(
        Signal.x1, Estimate.PERIODOGRAM, Mc, L, None, None, None, None, M, SIGMA_1, omega
    )
    estimator_periodogram_x2 = Estimator(
        Signal.x2, Estimate.PERIODOGRAM, Mc, L, None, None, None, None, M, SIGMA_2, omega
    )

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
