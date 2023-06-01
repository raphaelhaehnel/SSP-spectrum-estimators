import numpy as np
from enum import Enum
from signal_processing import *


class Estimator:
    """
    This class implements a specific estimator
    """

    def __init__(
        self,
        signal_type: Signal,
        estimator_type: Estimate,
        Mc: int,
        L: int,
        L_section: int,
        L_BT: int,
        K: int,
        D: int,
        M: int,
        sigma: float,
        omega: np.ndarray,
    ):
        self.signal_type = signal_type
        self.estimator_type = estimator_type
        self.Mc = Mc
        self.L = L
        self.L_section = L_section
        self.L_BT = L_BT
        self.K = K
        self.D = D
        self.M = M
        self.sigma = sigma
        self.omega = omega
        self.Sxx = generate_analytic_spectrum(self.signal_type, self.omega)

        self.mean = self.compute_mean()
        self.bias = self.compute_bias()
        self.variance = self.compute_variance()
        self.error = self.compute_mse()

    def compute_mean(self):
        mean = np.zeros(2 * self.L + 1)

        for i in range(self.Mc):
            # Generate the signal
            x = self.__get_signal()

            # Generate the estimation
            x_estimation = self.__get_estimation(x)

            # Add the current estimation to the mean of estimations
            mean += x_estimation

        return mean / self.Mc

    def compute_variance(self):
        variance = np.zeros(2 * self.L + 1)

        for i in range(self.Mc):
            # Generate the signal
            x = self.__get_signal()

            # Generate the estimation
            x_estimation = self.__get_estimation(x)

            # Add the current estimation to the mean of estimations
            variance += np.abs(x_estimation - self.mean) ** 2

        return variance / self.Mc

    def compute_bias(self):
        return self.mean - self.Sxx

    def compute_mse(self):
        return self.variance + self.bias**2

    def __get_signal(self):
        if self.signal_type == Signal.x1:
            return generate_x1_signal(2 * self.L + 1, self.sigma)
        elif self.signal_type == Signal.x2:
            return generate_x2_signal(2 * self.L + 1, self.sigma)
        else:
            error = f"Invalid parameter '{self.signal_type}'"
            raise ValueError(error)

    def __get_estimation(self, x: np.ndarray):
        if self.estimator_type == Estimate.PERIODOGRAM:
            return compute_periodogram(x, self.L, self.M)
        elif self.estimator_type == Estimate.BARTLETT:
            return compute_bartlett(x, self.M, self.K, self.L_section)
        elif self.estimator_type == Estimate.WELCH:
            return compute_welch(x, self.M, self.K, self.L_section, self.D)
        elif self.estimator_type == Estimate.BLACKMAN_TUKEY:
            return compute_BT(x, self.L, self.M, self.L_BT)
        else:
            error = f"Invalid parameter '{self.estimator_type}'"
            raise ValueError(error)
