"""Statistical metrics used to summarize ExoReL retrievals."""

import math
import os

import numpy as np
import scipy as sp
from spectres import spectres


def aic(max_log_likelihood, n_parameters):
    """Return the Akaike Information Criterion."""
    if max_log_likelihood is None or not np.isfinite(max_log_likelihood):
        return np.nan
    return float((2.0 * n_parameters) - (2.0 * max_log_likelihood))


def aicc(max_log_likelihood, n_parameters, n_data_points):
    """Return the sample-size-corrected Akaike Information Criterion."""
    value = aic(max_log_likelihood, n_parameters)
    if not np.isfinite(value) or n_data_points <= n_parameters + 1:
        return np.nan
    correction = (
        2.0 * n_parameters * (n_parameters + 1)
        / (n_data_points - n_parameters - 1)
    )
    return float(value + correction)


def bic(max_log_likelihood, n_parameters, n_data_points):
    """Return the Bayesian Information Criterion."""
    if (
        max_log_likelihood is None
        or not np.isfinite(max_log_likelihood)
        or n_data_points <= 0
    ):
        return np.nan
    return float(
        (np.log(n_data_points) * n_parameters)
        - (2.0 * max_log_likelihood)
    )


def bpics(log_likelihood_samples, n_parameters, weights=None):
    """Return the simplified Bayesian Predictive Information Criterion."""
    log_likelihood_samples = np.asarray(log_likelihood_samples, dtype=float)
    mean_log_likelihood = np.average(
        log_likelihood_samples,
        weights=weights,
    )
    if not np.isfinite(mean_log_likelihood):
        return None
    return float(
        (-2.0 * mean_log_likelihood)
        + (2.0 * float(n_parameters))
    )


def dic(log_likelihood_samples, weights=None):
    """Return the Ando/Gelman Deviance Information Criterion."""
    log_likelihood_samples = np.asarray(log_likelihood_samples, dtype=float)
    mean_log_likelihood = np.average(
        log_likelihood_samples,
        weights=weights,
    )
    if not np.isfinite(mean_log_likelihood):
        return None

    if weights is None:
        effective_parameters = 2.0 * np.var(log_likelihood_samples)
    else:
        variance = np.average(
            (log_likelihood_samples - mean_log_likelihood) ** 2,
            weights=weights,
        )
        effective_parameters = 2.0 * variance

    value = (
        -2.0 * mean_log_likelihood
        + 3.0 * effective_parameters
    )
    if not np.isfinite(value):
        return None
    return float(value)


def waic(pointwise_log_likelihood):
    """Return the widely applicable information criterion."""
    pointwise_log_likelihood = np.asarray(
        pointwise_log_likelihood,
        dtype=float,
    )
    if pointwise_log_likelihood.ndim != 2:
        return None
    if (
        pointwise_log_likelihood.shape[0] == 0
        or pointwise_log_likelihood.shape[1] == 0
    ):
        return None

    finite_columns = np.all(
        np.isfinite(pointwise_log_likelihood),
        axis=0,
    )
    pointwise_log_likelihood = pointwise_log_likelihood[:, finite_columns]
    if pointwise_log_likelihood.shape[1] == 0:
        return None

    n_samples = pointwise_log_likelihood.shape[0]
    fit_term = sp.special.logsumexp(
        pointwise_log_likelihood,
        axis=0,
        b=(1.0 / float(n_samples)),
    )
    penalty_term = np.var(pointwise_log_likelihood, axis=0)
    value = -2.0 * (
        np.sum(fit_term)
        - np.sum(penalty_term)
    )
    if not np.isfinite(value):
        return None
    return float(value)


def chi_square(data, model, error):
    """Return chi-square for a model evaluated at the data points."""
    residual = (data - model) / error
    return float(np.sum(residual ** 2.0))


def reduced_chi_square(chi_square_value, degrees_of_freedom):
    """Return chi-square divided by the degrees of freedom."""
    if degrees_of_freedom == 0 or not np.isfinite(chi_square_value):
        return np.nan
    return float(chi_square_value / degrees_of_freedom)


def sigma_from_log_evidence(log_evidence, reference_log_evidence):
    """Return Gaussian-equivalent separation from a reference evidence."""
    if (
        log_evidence is None
        or not np.isfinite(log_evidence)
        or not np.isfinite(reference_log_evidence)
    ):
        return None
    delta_log_evidence = reference_log_evidence - float(log_evidence)
    if delta_log_evidence <= 0.0:
        return None
    return float(np.sqrt(2.0 * delta_log_evidence))


def retrieval_n_data_points(param):
    """Return the total number of data points in a retrieval."""
    if param['obs_numb'] is None:
        return int(len(param['spectrum']['Fplanet']))

    n_data_points = 0
    for observation in range(int(param['obs_numb'])):
        n_data_points += len(
            param['spectrum'][str(observation)]['Fplanet']
        )
    return int(n_data_points)


def retrieval_log_likelihood_constant(param):
    """Return the Gaussian normalization term for retrieval data."""
    def validated_error_array(error):
        error = np.asarray(error, dtype=float)
        if error.size == 0:
            return np.array([], dtype=float)
        if not np.all(np.isfinite(error)) or np.any(error <= 0.0):
            return None
        return error

    normalization = np.sqrt(2.0 * math.pi)
    if param['obs_numb'] is None:
        error = validated_error_array(param['spectrum']['error_p'])
        if error is None:
            return np.nan
        return float(np.sum(np.log(error * normalization)))

    log_likelihood_constant = 0.0
    for observation in range(int(param['obs_numb'])):
        error = validated_error_array(
            param['spectrum'][str(observation)]['error_p']
        )
        if error is None:
            return np.nan
        log_likelihood_constant += float(
            np.sum(np.log(error * normalization))
        )
    return float(log_likelihood_constant)


def chi_square_from_best_fit_file(
    param,
    best_fit_path,
    spectral_binning,
):
    """Calculate chi-square from a saved best-fit spectrum."""
    if param['obs_numb'] is not None or not os.path.isfile(best_fit_path):
        return None

    best_fit = np.loadtxt(best_fit_path)
    if best_fit.ndim != 2 or best_fit.shape[1] < 2:
        return None

    model_wavelength = best_fit[:, 0]
    model_flux = best_fit[:, 1]
    if param['spectrum']['bins']:
        wavelength_bins = np.array(
            [
                param['spectrum']['wl_low'],
                param['spectrum']['wl_high'],
            ]
        ).T
        model_at_data = spectral_binning(
            wavelength_bins,
            model_wavelength,
            model_flux,
            bins=True,
        )
    else:
        model_at_data = spectres(
            param['spectrum']['wl'],
            model_wavelength,
            model_flux,
            fill=False,
        )

    return chi_square(
        param['spectrum']['Fplanet'],
        model_at_data,
        param['spectrum']['error_p'],
    )


def maximum_log_likelihood_from_chain_file(file_path):
    """Read the maximum log likelihood from a MultiNest chain file."""
    if not os.path.isfile(file_path):
        return None

    try:
        data = np.loadtxt(file_path)
    except (OSError, ValueError):
        return None

    if data.ndim == 1:
        if len(data) < 2:
            return None
        try:
            negative_twice_log_likelihood = np.array(
                [float(data[1])],
                dtype=float,
            )
        except (TypeError, ValueError):
            return None
    else:
        try:
            negative_twice_log_likelihood = np.asarray(
                data[:, 1],
                dtype=float,
            )
        except (TypeError, ValueError):
            return None

    finite = negative_twice_log_likelihood[
        np.isfinite(negative_twice_log_likelihood)
    ]
    if finite.size == 0:
        return None

    return float(-0.5 * np.min(finite))


def log_likelihood_samples_from_chain_file(file_path):
    """Read log-likelihood samples and normalized weights from a chain."""
    if not os.path.isfile(file_path):
        return None, None

    try:
        data = np.loadtxt(file_path)
    except (OSError, ValueError):
        return None, None

    if data.ndim == 1:
        if len(data) < 2:
            return None, None
        try:
            log_likelihood_samples = np.array(
                [-0.5 * float(data[1])],
                dtype=float,
            )
        except (TypeError, ValueError):
            return None, None
        weights = None
        try:
            sample_weight = float(data[0])
        except (TypeError, ValueError):
            sample_weight = np.nan
        if np.isfinite(sample_weight) and sample_weight > 0.0:
            weights = np.array([sample_weight], dtype=float)
    else:
        if data.shape[1] < 2:
            return None, None
        try:
            log_likelihood_samples = -0.5 * np.asarray(
                data[:, 1],
                dtype=float,
            )
        except (TypeError, ValueError):
            return None, None
        try:
            weights = np.asarray(data[:, 0], dtype=float)
        except (TypeError, ValueError):
            weights = None

    finite = np.isfinite(log_likelihood_samples)
    if weights is not None:
        finite &= np.isfinite(weights) & (weights > 0.0)

    log_likelihood_samples = log_likelihood_samples[finite]
    if log_likelihood_samples.size == 0:
        return None, None

    if weights is not None:
        weights = weights[finite]
        weight_sum = np.sum(weights)
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            weights = None
        else:
            weights = weights / weight_sum

    return log_likelihood_samples, weights


def bpics_from_chain_file(file_path, n_parameters):
    """Calculate BPICs from a MultiNest chain file."""
    samples, weights = log_likelihood_samples_from_chain_file(file_path)
    if samples is None:
        return None
    return bpics(samples, n_parameters, weights=weights)


def dic_from_chain_file(file_path):
    """Calculate DIC from a MultiNest chain file."""
    samples, weights = log_likelihood_samples_from_chain_file(file_path)
    if samples is None:
        return None
    return dic(samples, weights=weights)


def waic_from_pointwise_log_likelihood_file(
    file_path,
    n_expected_points=None,
):
    """Calculate WAIC from a saved pointwise log-likelihood array."""
    if not os.path.isfile(file_path):
        return None

    try:
        pointwise_log_likelihood = np.loadtxt(file_path)
    except (OSError, ValueError):
        return None

    pointwise_log_likelihood = np.asarray(
        pointwise_log_likelihood,
        dtype=float,
    )
    if pointwise_log_likelihood.ndim != 2:
        return None

    if n_expected_points is not None:
        if pointwise_log_likelihood.shape[1] == int(n_expected_points):
            pass
        elif pointwise_log_likelihood.shape[0] == int(n_expected_points):
            pointwise_log_likelihood = pointwise_log_likelihood.T

    return waic(pointwise_log_likelihood)


def maximum_log_likelihoods_from_post_separate(post_separate_path):
    """Read per-mode maximum log likelihoods from ``post_separate.dat``."""
    if not os.path.isfile(post_separate_path):
        return []

    mode_log_likelihoods = []
    current_minimum = None
    empty_rows = 0
    with open(post_separate_path, 'r') as post_separate_file:
        for index, raw_line in enumerate(post_separate_file):
            if index <= 2:
                continue

            line = raw_line.strip()
            if len(line) == 0:
                empty_rows += 1
                continue

            if empty_rows >= 2 and current_minimum is not None:
                mode_log_likelihoods.append(float(-0.5 * current_minimum))
                current_minimum = None
            empty_rows = 0

            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                negative_twice_log_likelihood = float(parts[1])
            except (TypeError, ValueError):
                continue
            if (
                current_minimum is None
                or negative_twice_log_likelihood < current_minimum
            ):
                current_minimum = negative_twice_log_likelihood

    if current_minimum is not None:
        mode_log_likelihoods.append(float(-0.5 * current_minimum))

    return mode_log_likelihoods
