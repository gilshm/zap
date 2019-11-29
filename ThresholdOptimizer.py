import numpy as np
from scipy.optimize import curve_fit


class Sigmoid:
    @staticmethod
    def run(x, a, b, c, d):
        calc = d + a / (1 + np.exp(-b * np.array(x) + c))
        return calc.tolist()


class ThresholdOptimizer:
    def __init__(self):
        self.error_curves = []
        self.mac_curves = []
        self._max_err = 0.5

    def curve_fit(self, x_data, y_data):
        """
        Find the best prior and parameters to curve fit.
        Notice, curve fit is not that stable, its performance may vary according to the x_data and y_data range.
        The user should inspect the output and modify the range via the Config.py file and/or the normalization
        in the main.py file.
        :param x_data: x values
        :param y_data: corresponding y values
        :return: pointer to the best prior found by curve fit, corresponding prior parameters
        """
        prior_list = [Sigmoid]  # Prior can be implemented and added
        best_prior = None
        best_params = None
        best_sse = np.inf

        for i, prior in enumerate(prior_list):
            try:
                popt, _ = curve_fit(prior.run, x_data, y_data, maxfev=16384)
                result = prior.run(x_data, *popt)
                sse = np.sum(np.power(np.array(y_data) - np.array(result), 2.0))

                if best_sse > sse > 0:
                    best_prior = prior
                    best_params = popt
                    best_sse = sse

            except Exception:
                pass

        return best_prior, best_params

    def set_max_err(self, err):
        self._max_err = err

    def _objective(self, x, layer):
        rec = self.mac_curves[layer]
        result = rec['prior'].run(x, *rec['params'])
        return result

    def _constraint(self, x, layer):
        rec = self.error_curves[layer]
        result = rec['prior'].run(x, *rec['params'])
        return result

    def objective(self, x):
        res = 0
        for i, _ in enumerate(self.mac_curves):
            res += self._objective(x[i], i)

        return res

    def constraint(self, x):
        res = 0
        for i, _ in enumerate(self.error_curves):
            res += self._constraint(x[i], i)

        return -res + self._max_err
