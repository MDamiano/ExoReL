"""Prior transformations used by ExoReL retrievals."""

import numpy as np
import scipy as sp
from scipy.interpolate import interp1d


def Mp_Rp_prior(param, parameter, cube, rp_value=None, mp_value=None):
    """
    Prior function for planetary mass and radius

    Parameters
    ----------
    param : dict
        dictionary of settings.
    cube : float
        Unit-cube value to be converted.
    parameter : str
        Parameter to evaluate. Choose between ``'Mp'`` and ``'Rp'``.
    rp_value : float, optional
        Radius value to be used in the Mass-Radius diagram.
    mp_value : float, optional
        Mass value to be used in the Mass-Radius diagram.


    Returns
    -------
    float
        Mass or radius value evaluated according to the requested prior.
    """

    if parameter == 'Mp':
        if rp_value is None:
            if param['Mp_err'] is None:
                return uniform_prior(param, 'Mp', cube)
            if param['Mp_prior_type'] == 'gaussian':
                return gaussian_prior(param, 'Mp', cube)
        else:
            return (cube * (param['M-R_Fe'](rp_value) - param['M-R_H2O'](rp_value))) + param['M-R_H2O'](rp_value)

    if parameter == 'Rp':
        if mp_value is None:
            if param['Rp_err'] is None:
                return uniform_prior(param, 'Rp', cube)
            if param['Rp_prior_type'] == 'gaussian':
                return gaussian_prior(param, 'Rp', cube)
        else:
            return (cube * (param['M-R_Fe'](mp_value) - param['M-R_H2O'](mp_value))) + param['M-R_H2O'](mp_value)

    raise ValueError("parameter must be either 'Mp' or 'Rp'")


def uniform_prior(param, parameter, cube):
    return (cube * (param[parameter + '_range'][1] - param[parameter + '_range'][0])) + param[parameter + '_range'][0]


def gaussian_prior(param, parameter, cube):
    range_array = np.linspace(param[parameter + '_range'][0], param[parameter + '_range'][1], num=10000, endpoint=True)
    cdf = sp.stats.norm.cdf(range_array, param[parameter + '_orig'], param[parameter + '_err'])
    cdf = np.array([0.0] + list(cdf) + [1.0])
    range_array = np.array([range_array[0]] + list(range_array) + [range_array[-1]])
    pri = interp1d(cdf, range_array)
    return pri(cube)


THERMAL_PARAMETERS = ('T_surface', 'T_join', 'tau_join', 'gamma', 'eta')


def _bounded_uniform(unit_value, lower, upper, description):
    from .thermodynamics import InvalidAtmosphereError
    u, lo, hi = float(unit_value), float(lower), float(upper)
    if not np.isfinite([u, lo, hi]).all() or not 0.0 <= u <= 1.0:
        raise InvalidAtmosphereError(f'Invalid unit coordinate or bounds for {description}.')
    if hi < lo:
        raise InvalidAtmosphereError(f'Empty conditional prior support for {description}.')
    return lo + u * (hi - lo)


def conditional_join_temperature(param, unit_value, T_surface):
    """Uniform T_join conditional on T_join <= T_surface (temperatures in K)."""
    lo, hi = param['T_join_range']
    return _bounded_uniform(unit_value, lo, min(hi, T_surface), 'T_join')


def coupled_thermal_prior(param, unit_values):
    """Transform five unit coordinates to thermal sampler coordinates.

    Order is THERMAL_PARAMETERS. T_surface, T_join, tau_join and eta are
    linear; gamma is log10. T_join has conditional support below T_surface.
    A tau_join range starting at zero retains the exact isothermal endpoint;
    zero has zero prior mass unless the parameter is fixed there.
    """
    unit_values = np.asarray(unit_values, dtype=float)
    if unit_values.shape != (5,):
        raise ValueError('The coupled thermal prior requires five coordinates.')
    result = np.empty(5)
    result[0] = _bounded_uniform(unit_values[0], *param['T_surface_range'], 'T_surface')
    result[1] = conditional_join_temperature(param, unit_values[1], result[0])
    for index, name in enumerate(THERMAL_PARAMETERS[2:], 2):
        result[index] = _bounded_uniform(unit_values[index], *param[name + '_range'], name)
    return result


def conditional_water_prior(param, unit_value, T_surface, P0=None,
                            gas_coordinates=None, clr_ppf=None):
    """Sample bottom water, returning its native gas coordinate.

    Below the critical point, enforce the saturation cap. Hot bottoms have
    no condensation cap; only the configured prior and composition apply.

    VMR mode: log-uniform water VMR conditional on temperature, total
    pressure and the other fitted gas VMRs, leaving positive filler gas when
    configured. A pure-water composition with no filler can reach q=1.
    Partial-pressure mode: log-uniform water partial pressure <= e_s(T).
    CLR mode: the original water-CLR marginal (uniform or tabulated inverse
    CDF) truncated at the saturation bound, conditional on all other fitted
    CLR coordinates. The implicit filler CLR is minus the sum of fitted
    CLRs. This keeps their coordinates fixed, not their ratios to the filler.
    No sampled abundance is clipped after decoding.
    """
    from scipy.optimize import brentq
    from scipy.special import logsumexp
    from .thermodynamics import InvalidAtmosphereError, vapor_pressure_limit

    space = param['gas_par_space']
    coordinates = dict(gas_coordinates or {})
    coordinates.pop('H2O', None)
    es = float(vapor_pressure_limit(float(T_surface)))
    if space == 'partial_pressure':
        lo, hi = param['ppH2O_range']
        return _bounded_uniform(unit_value, lo, min(hi, np.log10(es)), 'water partial pressure')

    pressure = float(param.get('P0') if P0 is None else P0)
    if not np.isfinite(pressure) or pressure <= 0.0:
        raise InvalidAtmosphereError('A positive pressure is required for the water prior.')
    qmax = min(1.0, es / pressure)
    if space in ('vmr', 'volume_mixing_ratio'):
        dry_total = sum(10.0 ** float(v) for v in coordinates.values())
        # Strictly positive filling gas is required only when configured.
        remaining = 1.0 - dry_total
        if param.get('gas_fill') is not None:
            remaining *= 1.0 - 1.0e-12
        qmax = min(qmax, remaining)
        if qmax <= 0.0:
            raise InvalidAtmosphereError('Other gases leave no water/filler prior support.')
        lo, hi = param['vmrH2O_range']
        return _bounded_uniform(unit_value, lo, min(hi, np.log10(qmax)), 'water VMR')

    if space not in ('clr', 'centered_log_ratio'):
        raise ValueError(f'Unsupported gas parameter space {space!r}.')
    others = np.asarray(list(coordinates.values()), dtype=float)
    if not np.isfinite(others).all():
        raise InvalidAtmosphereError('CLR coordinates must be finite.')
    if clr_ppf is None:
        lo, hi = map(float, param['clrH2O_range'])
    else:
        table = np.asarray(clr_ppf, dtype=float)
        if (table.ndim != 2 or table.shape[1] != 2 or len(table) < 2
                or not np.isfinite(table).all()
                or np.any(np.diff(table[:, 0]) <= 0.0)
                or np.any(np.diff(table[:, 1]) < 0.0)
                or table[0, 0] != 0.0 or table[-1, 0] != 1.0):
            raise ValueError('CLR inverse-CDF table must be monotone and span probabilities 0 to 1.')
        lo, hi = float(table[0, 1]), float(table[-1, 1])

    def log_water(z):
        return z - logsumexp(np.r_[others, z, -z - others.sum()])

    log_cap = np.log(qmax)
    if log_water(lo) > log_cap + 1.0e-13:
        raise InvalidAtmosphereError('Empty conditional water CLR support.')
    if log_water(hi) > log_cap:
        hi = brentq(lambda z: log_water(z) - log_cap, lo, hi, xtol=1.0e-13)
    if clr_ppf is None:
        return _bounded_uniform(unit_value, lo, hi, 'water CLR')
    # Restrict and renormalize the supplied marginal, rather than replacing
    # the modified CLR prior with a log-VMR prior.
    cdf_upper = float(np.interp(hi, table[:, 1], table[:, 0]))
    probability = _bounded_uniform(unit_value, 0.0, cdf_upper, 'water CLR probability')
    return float(np.interp(probability, table[:, 0], table[:, 1]))
