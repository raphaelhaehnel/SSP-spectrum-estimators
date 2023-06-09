import time
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

# Array of the names of the estimators we'll use with the Monte-Carlo
LIST_ESTIMATORS: np.ndarray = np.array(["periodogram", "bartlett16", "bartlett64", "welch_61", "welch_253", "bt_4", "bt_2"])


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
    plt.legend(loc="upper right")
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
    title: str,
    k_half: np.ndarray,
    x1_correlogram: np.ndarray,
    x1_periodogram: np.ndarray,
    x1_bartlett16: np.ndarray,
    x1_bartlett64: np.ndarray,
    x1_welch_61: np.ndarray,
    x1_welch_253: np.ndarray,
    x1_bt_4: np.ndarray,
    x1_bt_2: np.ndarray,
    Sxx1: np.ndarray,
    x2_correlogram: np.ndarray,
    x2_periodogram: np.ndarray,
    x2_bartlett16: np.ndarray,
    x2_bartlett64: np.ndarray,
    x2_welch_61: np.ndarray,
    x2_welch_253: np.ndarray,
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
    fig.suptitle(title)

    # if x1_correlogram is not None:
    #     ax[0].plot(k_half, x1_correlogram, label="Correlogram")

    # Display all the estimators for x1
    ax[0].plot(k_half, x1_periodogram, label="Periodogram")
    ax[0].plot(k_half, x1_bartlett16, label="Bartlet16")
    ax[0].plot(k_half, x1_bartlett64, label="Bartlet64")
    ax[0].plot(k_half, x1_welch_61, label="Welch61")
    ax[0].plot(k_half, x1_welch_253, label="Welch253")
    ax[0].plot(k_half, x1_bt_4, label="Blackman Tukey4")
    ax[0].plot(k_half, x1_bt_2, label="Blackman Tukey2")

    if Sxx1 is not None:
        ax[0].plot(k_half, Sxx1, label="Analytic")

    ax[0].set_xlabel(r"$\omega$")
    ax[0].set_ylabel(r"$\hat{S}_{XX_1}$")
    ax[0].legend(loc="upper right")
    ax[0].grid()

    # if x2_correlogram is not None:
    #     ax[1].plot(k_half, x2_correlogram, label="Correlogram")

    # Display all the estimators for x2
    ax[1].plot(k_half, x2_periodogram, label="Periodogram")
    ax[1].plot(k_half, x2_bartlett16, label="Bartlet16")
    ax[1].plot(k_half, x2_bartlett64, label="Bartlet64")
    ax[1].plot(k_half, x2_welch_61, label="Welch61")
    ax[1].plot(k_half, x2_welch_253, label="Welch253")
    ax[1].plot(k_half, x2_bt_4, label="Blackman Tukey4")
    ax[1].plot(k_half, x2_bt_2, label="Blackman Tukey2")

    if Sxx2 is not None:
        ax[1].plot(k_half, Sxx2, label="Analytic")

    ax[1].set_xlabel(r"$\omega$")
    ax[1].set_ylabel(r"$\hat{S}_{XX_2}$")
    ax[1].legend(loc="upper right")
    ax[1].grid()


def display_bar_chart(title: str, names: np.ndarray, x1: np.ndarray, x2: np.ndarray):
    """
    This function displays the performances values of the different estimators

    Parameters
    ----------
    title : The name of the graph
    names : Array containing the names of the different estimators
    x1 : Array of samples of signal x1
    x2 : Array of samples of signal x2
    """
    n_signals = 2

    # Figure Size
    fig, ax = plt.subplots(2, sharex=True)

    # Horizontal Bar Plot
    ax[0].barh(names, x1, color="tab:blue")
    ax[1].barh(names, x2, color="tab:orange")

    for j in range(n_signals):
        # Remove x, y Ticks
        ax[j].xaxis.set_ticks_position("none")
        ax[j].yaxis.set_ticks_position("none")

        # Add padding between axes and labels
        ax[j].xaxis.set_tick_params(pad=5)
        ax[j].yaxis.set_tick_params(pad=10)

        # Add x, y gridlines
        ax[j].grid(color="grey", linestyle="-.", linewidth=0.5, alpha=0.2)

        # Show top values
        ax[j].invert_yaxis()

        # Add annotation to bars
        for i in ax[j].patches:
            ax[j].text(
                i.get_width() + np.max([x2, x1]) / 100,
                i.get_y() + 0.5,
                str(round((i.get_width()), 2)),
                fontsize=10,
                fontweight="bold",
                color="grey",
            )

    # Add Plot Title
    fig.suptitle(title)


if __name__ == "__main__":
    start = time.time()

    # Question 1 #

    # For the first question, we want to display the result of the analytic computation of the spectrum
    # for x_1 and x_2

    # Resolution of the spectrum for the analytic signal
    L_frequencies = 2049

    # Defining omega as x asis
    omega = np.linspace(0, np.pi, L_frequencies)

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

    # Due to the nature of the DFT that we display only positive frequency, we need to ignore
    # half of the signal that is only a duplication of the spectrum
    k_half = np.linspace(0, np.pi, int(M / 2) + 1)

    end = time.time()

    print(f"Time | Basic computations : {'{:.2f}'.format(end-start)} sec")

    start = time.time()

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

    # Computation of the Welch estimation with 61 sections and overlapping of 48
    x1_welch_61 = compute_welch(x1, M, K=61, L_section=64, D=64 - 48)
    x2_welch_61 = compute_welch(x2, M, K=61, L_section=64, D=64 - 48)

    # Computation of the Welch estimation with 253 sections and overlapping of 12
    x1_welch_253 = compute_welch(x1, M, K=253, L_section=16, D=16 - 12)
    x2_welch_253 = compute_welch(x2, M, K=253, L_section=16, D=16 - 12)

    # Computation of the Blackman Tukey estimation with a window of 4
    x1_bt_4 = compute_BT(x1, L, M, L_BT=4)
    x2_bt_4 = compute_BT(x2, L, M, L_BT=4)

    # Computation of the Blackman Tukey estimation with a window of 2
    x1_bt_2 = compute_BT(x1, L, M, L_BT=2)
    x2_bt_2 = compute_BT(x2, L, M, L_BT=2)

    end = time.time()

    print(f"Time | Estimators : {'{:.2f}'.format(end-start)} sec")

    start = time.time()

    # The number of simulation (Monte-carlo)
    Mc = 100

    # Monte-Carlo for Periodogram
    mc_x1_periodogram = Estimator(Signal.x1, Estimate.PERIODOGRAM, Mc, SIGMA_1, omega, L, None, None, None, None, M)
    mc_x2_periodogram = Estimator(Signal.x2, Estimate.PERIODOGRAM, Mc, SIGMA_2, omega, L, None, None, None, None, M)

    # Monte-Carlo for Bartlett16
    mc_x1_bartlett_16 = Estimator(Signal.x1, Estimate.BARTLETT, Mc, SIGMA_1, omega, L, 16, 64, None, None, M)
    mc_x2_bartlett_16 = Estimator(Signal.x2, Estimate.BARTLETT, Mc, SIGMA_2, omega, L, 16, 64, None, None, M)

    # Monte-Carlo for Bartlett64
    mc_x1_bartlett_64 = Estimator(Signal.x1, Estimate.BARTLETT, Mc, SIGMA_1, omega, L, 64, 16, None, None, M)
    mc_x2_bartlett_64 = Estimator(Signal.x2, Estimate.BARTLETT, Mc, SIGMA_2, omega, L, 64, 16, None, None, M)

    # Monte-Carlo for Welch61
    mc_x1_welch_61 = Estimator(Signal.x1, Estimate.WELCH, Mc, SIGMA_1, omega, L, 61, 64, 64 - 48, None, M)
    mc_x2_welch_61 = Estimator(Signal.x2, Estimate.WELCH, Mc, SIGMA_2, omega, L, 61, 64, 64 - 48, None, M)

    # Monte-Carlo for Welch253
    mc_x1_welch_253 = Estimator(Signal.x1, Estimate.WELCH, Mc, SIGMA_1, omega, L, 253, 16, 16 - 12, None, M)
    mc_x2_welch_253 = Estimator(Signal.x2, Estimate.WELCH, Mc, SIGMA_2, omega, L, 253, 16, 16 - 12, None, M)

    # Monte-Carlo for BlackmanTukey4
    mc_x1_bt_4 = Estimator(Signal.x1, Estimate.BLACKMAN_TUKEY, Mc, SIGMA_1, omega, L, None, None, None, 4, M)
    mc_x2_bt_4 = Estimator(Signal.x2, Estimate.BLACKMAN_TUKEY, Mc, SIGMA_2, omega, L, None, None, None, 4, M)

    # Monte-Carlo for BlackmanTukey2
    mc_x1_bt_2 = Estimator(Signal.x1, Estimate.BLACKMAN_TUKEY, Mc, SIGMA_1, omega, L, None, None, None, 2, M)
    mc_x2_bt_2 = Estimator(Signal.x2, Estimate.BLACKMAN_TUKEY, Mc, SIGMA_2, omega, L, None, None, None, 2, M)

    end = time.time()

    print(f"Time | Monte-Carlo : {'{:.2f}'.format(end-start)} sec")

    start = time.time()

    # Displays the estimators WITHOUT monte-carlo
    display_estimators(
        "Estimators (Without Monte-Carlo)",
        k_half,
        x1_correlogram,
        x1_periodogram,
        x1_bartlett_16,
        x1_bartlett_64,
        x1_welch_61,
        x1_welch_253,
        x1_bt_4,
        x1_bt_2,
        Sxx1,
        x2_correlogram,
        x2_periodogram,
        x2_bartlett_16,
        x2_bartlett_64,
        x2_welch_61,
        x2_welch_253,
        x2_bt_4,
        x2_bt_2,
        Sxx2,
    )

    # Displays the estimators WITH monte-carlo
    display_estimators(
        "Estimators (With Monte-Carlo)",
        k_half,
        None,
        mc_x1_periodogram.mean,
        mc_x1_bartlett_16.mean,
        mc_x1_bartlett_64.mean,
        mc_x1_welch_61.mean,
        mc_x1_welch_253.mean,
        mc_x1_bt_4.mean,
        mc_x1_bt_2.mean,
        Sxx1,
        None,
        mc_x2_periodogram.mean,
        mc_x2_bartlett_16.mean,
        mc_x2_bartlett_64.mean,
        mc_x2_welch_61.mean,
        mc_x2_welch_253.mean,
        mc_x2_bt_4.mean,
        mc_x2_bt_2.mean,
        Sxx2,
    )

    # Displays the estimators bias
    display_estimators(
        "Bias",
        k_half,
        None,
        mc_x1_periodogram.bias,
        mc_x1_bartlett_16.bias,
        mc_x1_bartlett_64.bias,
        mc_x1_welch_61.bias,
        mc_x1_welch_253.bias,
        mc_x1_bt_4.bias,
        mc_x1_bt_2.bias,
        None,
        None,
        mc_x2_periodogram.bias,
        mc_x2_bartlett_16.bias,
        mc_x2_bartlett_64.bias,
        mc_x2_welch_61.bias,
        mc_x2_welch_253.bias,
        mc_x2_bt_4.bias,
        mc_x2_bt_2.bias,
        None,
    )

    # Displays the estimators variance
    display_estimators(
        "Variance",
        k_half,
        None,
        mc_x1_periodogram.variance,
        mc_x1_bartlett_16.variance,
        mc_x1_bartlett_64.variance,
        mc_x1_welch_61.variance,
        mc_x1_welch_253.variance,
        mc_x1_bt_4.variance,
        mc_x1_bt_2.variance,
        None,
        None,
        mc_x2_periodogram.variance,
        mc_x2_bartlett_16.variance,
        mc_x2_bartlett_64.variance,
        mc_x2_welch_61.variance,
        mc_x2_welch_253.variance,
        mc_x2_bt_4.variance,
        mc_x2_bt_2.variance,
        None,
    )

    # Displays the estimators error
    display_estimators(
        "MSE",
        k_half,
        None,
        mc_x1_periodogram.error,
        mc_x1_bartlett_16.error,
        mc_x1_bartlett_64.error,
        mc_x1_welch_61.error,
        mc_x1_welch_253.error,
        mc_x1_bt_4.error,
        mc_x1_bt_2.error,
        None,
        None,
        mc_x2_periodogram.error,
        mc_x2_bartlett_16.error,
        mc_x2_bartlett_64.error,
        mc_x2_welch_61.error,
        mc_x2_welch_253.error,
        mc_x2_bt_4.error,
        mc_x2_bt_2.error,
        None,
    )

    # Mean value for the bias of the Monte-Carlo for the signal x1
    x1_biases = np.array(
        [
            mc_x1_periodogram.bias_value,
            mc_x1_bartlett_16.bias_value,
            mc_x1_bartlett_64.bias_value,
            mc_x1_welch_61.bias_value,
            mc_x1_welch_253.bias_value,
            mc_x1_bt_4.bias_value,
            mc_x1_bt_2.bias_value,
        ]
    )

    # Mean value for the bias of the Monte-Carlo for the signal x2
    x2_biases = np.array(
        [
            mc_x2_periodogram.bias_value,
            mc_x2_bartlett_16.bias_value,
            mc_x2_bartlett_64.bias_value,
            mc_x2_welch_61.bias_value,
            mc_x2_welch_253.bias_value,
            mc_x2_bt_4.bias_value,
            mc_x2_bt_2.bias_value,
        ]
    )

    # Mean value for the variance of the Monte-Carlo for the signal x1
    x1_variance = np.array(
        [
            mc_x1_periodogram.variance_value,
            mc_x1_bartlett_16.variance_value,
            mc_x1_bartlett_64.variance_value,
            mc_x1_welch_61.variance_value,
            mc_x1_welch_253.variance_value,
            mc_x1_bt_4.variance_value,
            mc_x1_bt_2.variance_value,
        ]
    )

    # Mean value for the variance of the Monte-Carlo for the signal x2
    x2_variance = np.array(
        [
            mc_x2_periodogram.variance_value,
            mc_x2_bartlett_16.variance_value,
            mc_x2_bartlett_64.variance_value,
            mc_x2_welch_61.variance_value,
            mc_x2_welch_253.variance_value,
            mc_x2_bt_4.variance_value,
            mc_x2_bt_2.variance_value,
        ]
    )

    # Mean value for the mean square error of the Monte-Carlo for the signal x1
    x1_error = np.array(
        [
            mc_x1_periodogram.error_value,
            mc_x1_bartlett_16.error_value,
            mc_x1_bartlett_64.error_value,
            mc_x1_welch_61.error_value,
            mc_x1_welch_253.error_value,
            mc_x1_bt_4.error_value,
            mc_x1_bt_2.error_value,
        ]
    )

    # Mean value for the mean square error of the Monte-Carlo for the signal x2
    x2_error = np.array(
        [
            mc_x2_periodogram.error_value,
            mc_x2_bartlett_16.error_value,
            mc_x2_bartlett_64.error_value,
            mc_x2_welch_61.error_value,
            mc_x2_welch_253.error_value,
            mc_x2_bt_4.error_value,
            mc_x2_bt_2.error_value,
        ]
    )

    # Displays the mean value as bar chart for each metric, for each estimator and for each signal
    display_bar_chart("Bias values", LIST_ESTIMATORS, x1_biases, x2_biases)
    display_bar_chart("Variance values", LIST_ESTIMATORS, x1_variance, x2_variance)
    display_bar_chart("Error values", LIST_ESTIMATORS, x1_error, x2_error)

    end = time.time()

    print(f"Time | Displaying : {'{:.2f}'.format(end-start)} sec")

    plt.show()
