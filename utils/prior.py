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
