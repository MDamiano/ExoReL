"""Likelihood functions used by ExoReL retrievals."""

import math

import numpy as np


def gaussian_loglike(data, model, error):
    """Return the independent Gaussian log likelihood of a spectrum."""
    chi = (data - model) / error
    return (
        -np.sum(np.log(error * np.sqrt(2.0 * math.pi)))
        - 0.5 * np.sum(chi * chi)
    )
