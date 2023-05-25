import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import fft


def display_analytic_spectrum(omega: np.ndarray, Sxx1: np.ndarray, Sxx2: np.ndarray):
    plt.figure()
    plt.title("Analytic computation of the spectrum")
    plt.plot(omega, Sxx1, label="$S_{XX_1}$", color="tab:blue")
    plt.plot(omega, Sxx2, label="$S_{XX_2}$", color="tab:orange")
    plt.legend()
    plt.grid()
    plt.xlabel(r"$\omega$")
    plt.ylabel(r"$S_{XX}$")


def display_samples(x1: np.ndarray, x2: np.ndarray):
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


def generate_x2_initial(w_2, L):
    x2 = np.zeros(L)
    initial = np.random.randn()
    for i in range(L):
        if i == 0:
            x2[i] = 0.7 * initial + w_2[i]
        else:
            x2[i] = 0.7 * x2[i - 1] + w_2[i]
    return x2


def compute_periodogram(x):
    x_periodogram = 1 / L * abs(fft.fft(x, M)) ** 2
    x_periodogram_half = x_periodogram[: int(M / 2) + 1]

    return x_periodogram_half / 2


def compute_correlogram(x):
    Rx = 1 / L * np.correlate(x, x, mode="full")
    x_correlogram = np.abs(fft.fft(Rx, M))
    x_correlogram_half = x_correlogram[: int(M / 2) + 1]

    return x_correlogram_half / 2


def mean_periodogram_x1(n):
    mean_periodogram = np.zeros(2 * L + 1)

    for i in range(n):
        w_1 = sigma_1 * np.random.randn(2 * L + 1)
        x1 = signal.lfilter([1, -3, -4], [1], w_1)
        x1_periodogram = compute_periodogram(x1)
        mean_periodogram += x1_periodogram

    return mean_periodogram / n


def mean_periodogram_x2(n):
    mean_periodogram = np.zeros(2 * L + 1)

    for i in range(n):
        w_2 = sigma_2 * np.random.randn(2 * 2 * L + 1)
        x2 = signal.lfilter([1], [1, -0.7], w_2)
        x2 = x2[L:]
        x2_periodogram = compute_periodogram(x2)
        mean_periodogram += x2_periodogram

    return mean_periodogram / n / 1.5  # TODO why do I need to do this ??


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
L_2 = 2048
w_1 = sigma_1 * np.random.randn(L)
w_2 = sigma_2 * np.random.randn(2 * L)

x1 = signal.lfilter([1, -3, -4], [1], w_1)

x2 = signal.lfilter([1], [1, -0.7], w_2)
x2_initial = generate_x2_initial(w_2, L)

display_samples(x1, x2)

x2 = x2[L:]

M = 4096
k = np.linspace(0, 2 * np.pi, M)
k_half = np.linspace(0, np.pi, int(M / 2) + 1)


# Computation of the periodogram
x1_periodogram = compute_periodogram(x1)
x2_periodogram = compute_periodogram(x2)


# Computation of the Correlogram
x1_correlogram = compute_correlogram(x1)
x2_correlogram = compute_correlogram(x2)

# Monte-carlo simulation
x1_mean_period = mean_periodogram_x1(2000)
x2_mean_period = mean_periodogram_x2(2000)

fig, ax = plt.subplots(2, sharex=True)
fig.suptitle("Estimators")
ax[0].plot(k_half, x1_correlogram, color="tab:purple", label="Correlogram")
ax[0].plot(k_half, x1_periodogram, color="tab:green", label="Periodogram")
ax[0].plot(k_half, x1_mean_period, color="tab:blue", label="Mean")
ax[0].plot(k_half, Sxx1, color="tab:red", label="Analytic")
ax[0].set_xlabel(r"$\omega$")
ax[0].set_ylabel(r"$\hat{S}_{XX_1}$")
ax[0].legend()
ax[0].grid()

ax[1].plot(k_half, x2_correlogram, color="tab:purple", label="Correlogram")
ax[1].plot(k_half, x2_periodogram, color="tab:green", label="Periodogram")
ax[1].plot(k_half, x2_mean_period, color="tab:blue", label="Mean")
ax[1].plot(k_half, Sxx2, color="tab:red", label="Analytic")
ax[1].set_xlabel(r"$\omega$")
ax[1].set_ylabel(r"$\hat{S}_{XX_2}$")
ax[1].legend()
ax[1].grid()


plt.show()
