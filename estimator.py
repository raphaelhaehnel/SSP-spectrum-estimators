import numpy as np
from enum import Enum
from signal_processing import *


class Estimator:
    """
    This class implements an estimator for a specific signal and a specific estimator.
    It computes the Monte-Carlo, its variance, bias and error.
    """

    def __init__(
        self,
        signal_type: Signal,
        estimator_type: Estimate,
        Mc: int,
        sigma: float,
        omega: np.ndarray,
        L: int,
        K: int,
        L_section: int,
        D: int,
        L_BT: int,
        M: int,
    ):
        self.signal_type = signal_type
        self.estimator_type = estimator_type
        self.Mc = Mc
        self.L = L
        self.K = K
        self.L_section = L_section
        self.D = D
        self.L_BT = L_BT
        self.M = M
        self.sigma = sigma
        self.omega = omega
        self.Sxx = generate_analytic_spectrum(self.signal_type, self.omega)

        self.mean = self.compute_mean()
        self.bias = self.compute_bias()
        self.variance = self.compute_variance()
        self.error = self.compute_mse()

        # For each metric, we are computed its mean value. It isn't a statistical mean, but a mean over different frequencies
        self.bias_value = np.mean(self.bias**2)
        self.variance_value = np.mean(self.variance)
        self.error_value = np.mean(self.error)

    def compute_mean(self):
        """
        Computes the mean (Monte-Carlo) over multiple simulations and their estimations

        Returns
        -------
        mean : The mean estimation
        """

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
        """
        Computes the variance of the Monte-Carlo

        Returns
        -------
        variance : the Monte-Carlo variance
        """

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
        """
        Computes the bias of the Monte-Carlo

        Returns
        -------
        bias : the Monte-Carlo bias
        """

        return self.mean - self.Sxx

    def compute_mse(self):
        """
        Computes the mean square error of the Monte-Carlo

        Returns
        -------
        mse : the mean square error of the Monte-Carlo
        """

        return self.variance + self.bias**2

    def __get_signal(self):
        """
        Returns samples of the appropriate signal according to the variable signal_type

        Returns
        -------
        x : Array of samples of a signal
        """

        if self.signal_type == Signal.x1:
            return generate_x1_signal(self.L, self.sigma)
        elif self.signal_type == Signal.x2:
            return generate_x2_signal(self.L, self.sigma)
        else:
            error = f"Invalid parameter '{self.signal_type}'"
            raise ValueError(error)

    def __get_estimation(self, x: np.ndarray):
        """
        Returns the appropriate estimation according to the variable estimator_type

        Parameters
        ----------
        x : Array of samples of a signal

        Returns
        -------
        estimation : Spectrum estimation of the signal
        """

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
