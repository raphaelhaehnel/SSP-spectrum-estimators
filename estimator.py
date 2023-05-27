import numpy as np


class Estimator:
    """
    This class implements a specific estimator
    """

    def __init__(self, Mc, L, M, signal, Sxx, sigma):
        self.mean = self.__mean_periodogram(Mc, L, M, signal, sigma)
        self.bias = self.mean - Sxx
        self.variance = self.__variance_periodogram(Mc, L, M, signal, sigma)
        self.error = self.__MSE()

    def __mean_periodogram(self, Mc, L, M, signal, sigma):
        from main import generate_x1_signal, generate_x2_signal, compute_periodogram

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

    def __variance_periodogram(self, Mc, L, M, signal, sigma):
        from main import generate_x1_signal, generate_x2_signal, compute_periodogram

        mean_periodogram = np.zeros(2 * L + 1)

        for i in range(Mc):
            x = (
                generate_x1_signal(2 * L + 1, sigma)
                if signal == "x1"
                else generate_x2_signal(2 * L + 1, sigma)
            )
            x_periodogram = compute_periodogram(x, L, M)
            mean_periodogram += np.abs(x_periodogram - self.mean) ** 2

        return mean_periodogram / Mc

    def __MSE(self):
        return self.variance + self.bias**2
