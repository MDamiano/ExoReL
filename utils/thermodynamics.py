"""Saturation pressures and coupled parametric temperature profiles.

All pressures are total gas pressures in Pa, temperatures K, and abundances
molar/volume fractions. This is a temperature-continuous retrieval model, not
a radiative-convective equilibrium or buoyancy/stability solver. Only H2O
latent heating enters the thermal gradient; other species form a dry mixture.
T_surface is the temperature at the lower boundary P0, which need not be a
solid surface. At/above the water critical temperature the convective branch
is unsaturated and uses the same fixed-composition ideal-gas heat capacity.

Species molar Cp(T), in J mol^-1 K^-1, uses NIST WebBook Shomate fits,
linearly interpolated NIST-JANAF water and Gurvich methane tables, and 5R/2
for He. Source URLs, temperature intervals and coefficients are stored in
forward_mod/Data/heat_capacity_data.json. Outside each species' data range Cp
is held at the nearest endpoint; the fits are never extrapolated. Mixtures
are weighted for each sampled composition. Constant per-species overrides
remain available through param['molar_heat_capacities'] in the same units.

Saturation uses IAPWS liquid water and Murphy & Koop (2005), Eq.7 ice:
https://iapws.org/relguide/Supp-sat.html ; doi:10.1256/qj.04.94.
L_m = R T^2 d(log(e_s))/dT is the effective ideal-gas CC latent heat;
it is not a real-steam EOS latent heat near the critical point. The moist
gradient is the nondilute pseudoadiabat (Ding & Pierrehumbert 2016, Eq.12,
doi:10.3847/0004-637X/822/1/24), with immediate removal of condensate.
"""
from __future__ import annotations

import numpy as np
import math
import json
from bisect import bisect_right
from functools import lru_cache
from pathlib import Path
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.special import expn

class InvalidAtmosphereError(ValueError):
    """A sampled thermodynamic state lies outside the model domain."""


R = 8.31446261815324
T_TRIPLE = 273.16
T_CRITICAL = 647.096
P_CRITICAL = 22.064e6
_A = np.array([-7.85951783, 1.84408259, -11.7866497, 22.6807411,
               -15.9618719, 1.80122502])
_B = np.array([1., 1.5, 3., 3.5, 4., 7.5])
_COEFFICIENTS = tuple(zip(_A.tolist(), _B.tolist()))


class _HeatCapacity:
    """Piecewise Cp=A+B*t+C*t²+D*t³+E/t², t=T/1000, and its Cp/T integral.

    The first and last segments are constant endpoint extensions. Integral
    offsets make entropy continuous even when rounded source fits have a
    small Cp mismatch at an internal boundary. Those fits are not altered.
    """

    def __init__(self, breaks, coefficients):
        self.breaks = tuple(map(float, breaks))
        self.coefficients = tuple(tuple(map(float, row)) for row in coefficients)
        if (len(self.coefficients) != len(self.breaks) + 1
                or any(len(row) != 5 for row in self.coefficients)
                or not np.isfinite(self.coefficients).all()
                or not np.isfinite(self.breaks).all()
                or any(t <= 0 for t in self.breaks)
                or any(b <= a for a, b in zip(self.breaks, self.breaks[1:]))):
            raise InvalidAtmosphereError('Invalid heat-capacity intervals.')
        self.log_breaks = tuple(map(math.log, self.breaks))
        offsets = [0.]
        for i, y in enumerate(self.log_breaks):
            offsets.append(offsets[-1] + self._primitive(y, self.coefficients[i])
                           - self._primitive(y, self.coefficients[i+1]))
        self.offsets = tuple(offsets)

    @staticmethod
    def _value(T, coefficients):
        a, b, c, d, e = coefficients
        t = T / 1000.
        return a + t*(b + t*(c + t*d)) + e/(t*t)

    @staticmethod
    def _primitive(y, coefficients):
        a, b, c, d, e = coefficients
        if b == c == d == e == 0.:
            return a*y
        t = math.exp(y) / 1000.
        return a*y + b*t + c*t*t/2 + d*t*t*t/3 - e/(2*t*t)

    @classmethod
    def constant(cls, value):
        try:
            value = float(value)
        except (TypeError, ValueError) as error:
            raise InvalidAtmosphereError('Gas molar Cp must be a finite scalar greater than R.') from error
        if not math.isfinite(value) or value <= R:
            raise InvalidAtmosphereError('Gas molar Cp must be finite and greater than R.')
        return cls((), [(value, 0., 0., 0., 0.)])

    @classmethod
    def mix(cls, components):
        components = [(weight, model) for weight, model in components if weight > 0]
        breaks = sorted({t for _, model in components for t in model.breaks})
        probes = [breaks[0]/2, *breaks] if breaks else [298.15]
        coefficients = [sum((weight*np.asarray(model.coefficients[bisect_right(model.breaks, T)])
                             for weight, model in components), np.zeros(5)) for T in probes]
        return cls(breaks, coefficients)

    def scalar(self, T):
        coefficients = self.coefficients[bisect_right(self.breaks, T)]
        if coefficients[1:] == (0., 0., 0., 0.):
            return coefficients[0]
        return self._value(T, coefficients)

    def __call__(self, T):
        T = np.asarray(T, dtype=float)
        if np.any(~np.isfinite(T)) or np.any(T <= 0):
            raise InvalidAtmosphereError('Heat-capacity temperature must be finite and positive.')
        if T.ndim == 0:
            return self.scalar(float(T))
        return np.asarray([self.scalar(float(t)) for t in T.flat]).reshape(T.shape)

    def entropy_log(self, y):
        """Continuous primitive of Cp(T)/T, given log(T); arbitrary zero."""
        index = bisect_right(self.log_breaks, y)
        return self._primitive(y, self.coefficients[index]) + self.offsets[index]


@lru_cache(maxsize=1)
def _heat_capacity_data():
    data_path = Path(__file__).resolve().parents[1] / 'forward_mod' / 'Data' / 'heat_capacity_data.json'
    with data_path.open() as stream:
        return json.load(stream)


@lru_cache(maxsize=None)
def _species_heat_capacity(species):
    try:
        data = _heat_capacity_data()[species]
    except KeyError as error:
        raise InvalidAtmosphereError('Supply molar heat capacity for ' + species) from error
    if data['kind'] == 'constant':
        return _HeatCapacity.constant(data['cp_over_R'] * R)
    if data['kind'] == 'linear_table':
        points = np.asarray(data['points'])
        breaks = points[:, 0]
        slopes = np.diff(points[:, 1]) / np.diff(breaks)
        coefficients = [(cp - slope*T, slope*1000., 0., 0., 0.)
                        for (T, cp), slope in zip(points[:-1], slopes)]
    else:
        intervals = data['intervals']
        if any(a[1] != b[0] for a, b in zip(intervals, intervals[1:])):
            raise InvalidAtmosphereError('Heat-capacity fits must cover contiguous intervals.')
        breaks = [row[0] for row in intervals] + [intervals[-1][1]]
        coefficients = [row[2:] for row in intervals]
    low = _HeatCapacity._value(breaks[0], coefficients[0])
    high = _HeatCapacity._value(breaks[-1], coefficients[-1])
    return _HeatCapacity(breaks, [(low, 0., 0., 0., 0.), *coefficients,
                                 (high, 0., 0., 0., 0.)])


def species_heat_capacity(species, T):
    """Molar Cp(T) with constant endpoint extensions, in J mol^-1 K^-1."""
    return _species_heat_capacity(species)(T)


def _as_heat_capacity(value, default_species):
    if value is None:
        return _species_heat_capacity(default_species)
    return value if isinstance(value, _HeatCapacity) else _HeatCapacity.constant(value)


def _liquid_log_pressure(T):
    theta = 1.0 - T / T_CRITICAL
    return np.log(P_CRITICAL) + T_CRITICAL / T * np.sum(
        _A * theta[..., None] ** _B, axis=-1)


def _ice_log_pressure(T):
    return 9.550426 - 5723.265 / T + 3.53068 * np.log(T) - .00728332 * T


# Remove the tiny mismatch between the independently rounded empirical fits.
_ICE_OFFSET = float(_liquid_log_pressure(np.asarray(T_TRIPLE))
                    - _ice_log_pressure(np.asarray(T_TRIPLE)))


def _scalar_sat_latent(T, phase=None):
    """Allocation-free saturation and effective latent heat for ODE stages."""
    if phase == 'ice' or (phase is None and T < T_TRIPLE):
        loge = 9.550426 - 5723.265/T + 3.53068*math.log(T) - .00728332*T + _ICE_OFFSET
        L = R*(5723.265 + 3.53068*T - .00728332*T*T)
    else:
        theta = 1. - T/T_CRITICAL
        S = sum(a*theta**b for a,b in _COEFFICIENTS)
        dS = sum(a*b*theta**(b-1) for a,b in _COEFFICIENTS)
        loge = math.log(P_CRITICAL) + T_CRITICAL/T*S
        L = R*(-T_CRITICAL*S - T*dS)
    return math.exp(loge), L


def _valid_temperature(T):
    T = np.asarray(T, dtype=float)
    if np.any(~np.isfinite(T)) or np.any(T < 110) or np.any(T >= T_CRITICAL):
        raise InvalidAtmosphereError('Temperature must be finite and in [110, 647.096) K.')
    return T


def _valid_atmospheric_temperature(T):
    """Atmospheric domain, independent of a condensable's critical point."""
    T = np.asarray(T, dtype=float)
    if np.any(~np.isfinite(T)) or np.any(T < 110.):
        raise InvalidAtmosphereError('Atmospheric temperature must be finite and >= 110 K.')
    return T


def water_saturation_pressure(T):
    """Water saturation pressure (Pa) for scalar/array T in [110, 647.096) K.

    Uses stable ice below 273.16 K and liquid water at/above it.
    """
    T = _valid_temperature(T)
    # Both fits are safe on this interval, although only the selected one is used.
    return np.exp(np.where(T < T_TRIPLE, _ice_log_pressure(T) + _ICE_OFFSET,
                           _liquid_log_pressure(T)))


AMMONIA_T_TRIPLE = 195.495
AMMONIA_T_CRITICAL = 405.56
AMMONIA_P_CRITICAL = 11.365e6
_AMMONIA_A = np.array([-7.2257, 1.4263, -0.59642, -2.798, -3.7869])
_AMMONIA_B = np.array([1., 1.5, 2., 3.6, 15.5])


def _ammonia_liquid_log_pressure(T):
    theta = 1. - T / AMMONIA_T_CRITICAL
    return np.log(AMMONIA_P_CRITICAL) + AMMONIA_T_CRITICAL / T * np.sum(
        _AMMONIA_A * theta[..., None] ** _AMMONIA_B, axis=-1)


def _ammonia_solid_log_pressure(T):
    # NBS Report 9043, section 2.2, Eq. (1): log10(P/Torr).
    return np.log(101325. / 760.) + np.log(10.) * (9.98379 - 1627.22 / T)


_AMMONIA_SOLID_OFFSET = float(
    _ammonia_liquid_log_pressure(np.asarray(AMMONIA_T_TRIPLE))
    - _ammonia_solid_log_pressure(np.asarray(AMMONIA_T_TRIPLE)))


def ammonia_saturation_pressure(T):
    """Pure-ammonia saturation pressure (Pa), preserving scalar/array shape.

    Accepts finite T in [110, 405.56) K; invalid temperatures raise
    InvalidAtmosphereError. Solid NH3 is used below 195.495 K, liquid above.
    The liquid vapor-pressure ancillary is the Gao EOS fit used by CoolProp:
    https://github.com/CoolProp/CoolProp/blob/master/dev/fluids/Ammonia.json
    (ANCILLARIES.pS; 195.495--405.56 K).

    The solid fit is NBS Report 9043 (1966), section 2.2, Eq. (1):
    https://nvlpubs.nist.gov/nistpubs/Legacy/RPT/nbsreport9043.pdf
    Its tabulated range is 173--195.46 K. Below 173 K this is an explicit
    extrapolation down to the atmospheric model's 110 K lower limit, not
    a validated low-temperature fit. A constant log-pressure offset joins
    it continuously to the liquid fit at the adopted triple point.
    """
    T = np.asarray(T, dtype=float)
    if np.any(~np.isfinite(T)) or np.any(T < 110.) or np.any(T >= AMMONIA_T_CRITICAL):
        raise InvalidAtmosphereError(
            'Ammonia temperature must be finite and in [110, 405.56) K.')
    # Evaluate the liquid fit only within its domain, including for arrays
    # containing both solid and liquid temperatures.
    liquid = _ammonia_liquid_log_pressure(np.maximum(T, AMMONIA_T_TRIPLE))
    solid = _ammonia_solid_log_pressure(T) + _AMMONIA_SOLID_OFFSET
    pressure = np.exp(np.where(T < AMMONIA_T_TRIPLE, solid, liquid))
    return pressure.item() if pressure.shape == () else pressure


def vapor_pressure_limit(T, species='H2O'):
    """Condensation cap (Pa), with no cap at/above the critical temperature.

    Infinity is a sentinel for the noncondensing ideal-gas branch, NOT an
    extrapolated saturation pressure. Evaluate phase laws only below their
    critical points. The atmospheric lower temperature bound remains 110 K.
    """
    T = _valid_atmospheric_temperature(T)
    if species == 'H2O':
        critical, saturation = T_CRITICAL, water_saturation_pressure
    elif species == 'NH3':
        critical, saturation = AMMONIA_T_CRITICAL, ammonia_saturation_pressure
    else:
        raise ValueError('Unsupported condensable: ' + species)
    pressure = np.full(T.shape, np.inf)
    cold = T < critical
    if np.any(cold):
        pressure[cold] = saturation(T[cold])
    return pressure.item() if pressure.shape == () else pressure


def latent_molar(T):
    """Effective L=R*T**2*d(log(es))/dT; ideal-CC consistent, not real-EOS L."""
    T = _valid_temperature(T)
    theta = 1.0 - T / T_CRITICAL
    S = np.sum(_A * theta[..., None] ** _B, axis=-1)
    dS = np.sum(_A * _B * theta[..., None] ** (_B - 1), axis=-1)
    liquid = -T_CRITICAL / T**2 * S - dS / T
    ice = 5723.265 / T**2 + 3.53068 / T - .00728332
    return R * T**2 * np.where(T < T_TRIPLE, ice, liquid)


def moist_gradient(T, q, cp_d=None, cp_v=None):
    """Nondilute saturated pseudoadiabatic gradient, regular at q=0 and q=1."""
    T = np.asarray(T, dtype=float)
    q = np.asarray(q, dtype=float)
    if np.any((q < 0) | (q > 1)):
        raise InvalidAtmosphereError('Water VMR must be in [0, 1].')
    cp_d = _as_heat_capacity(cp_d, 'N2')(T)
    cp_v = _as_heat_capacity(cp_v, 'H2O')(T)
    ell = latent_molar(T) / (R * T)
    cp = (1 - q) * cp_d + q * cp_v
    return R * ((1 - q) + ell * q) / ((1 - q) * cp + R * ell**2 * q)


def convection_profile(P, Tsurf, qdeep, Tjoin=200., P0=1.e5, method='fast',
                       max_step=0.02, cp_d=None, cp_v=None):
    """Lower-boundary-anchored dry/moist profile with an isothermal cap.

    P is a 1D array in either order, with 0<P<=P0. Output order matches input.
    Pjoin is None if the cap is not reached by min(P). qdeep is the actual
    bottom water VMR and must not exceed saturation at a subcritical bottom.
    Hot layers follow the unsaturated adiabat until a subcritical saturation
    crossing. A trajectory already supersaturated upon entering that domain
    cannot join continuously and requires nonideal phase physics; it raises
    InvalidAtmosphereError. 'fast' uses RK4
    in log pressure; 'reference' integrates log pressure versus log temperature
    using adaptive DOP853 and root-inverts the independent-variable solution.
    Both use the same thermodynamic assumptions but different integration paths.
    cp_d/cp_v accept heat-capacity models or constant molar overrides; defaults
    are temperature-dependent N2/H2O. Unsaturated motion uses the exact
    piecewise integral of Cp(T)/T. Moist stages evaluate Cp at the local T.
    """
    P = np.asarray(P, dtype=float)
    if method not in ('fast', 'reference'):
        raise InvalidAtmosphereError("method must be 'fast' or 'reference'.")
    Tsurf, Tjoin, qdeep, P0 = map(float, (Tsurf, Tjoin, qdeep, P0))
    _valid_atmospheric_temperature([Tsurf, Tjoin])
    if P.ndim != 1 or not len(P) or np.any(~np.isfinite(P)) or np.any(P <= 0):
        raise InvalidAtmosphereError('P must be a nonempty finite positive 1D array.')
    if not np.isfinite(P0) or P0 <= 0 or np.any(P > P0 * (1 + 1.e-12)):
        raise InvalidAtmosphereError('All pressure points must lie at or above the surface.')
    if not 0 <= qdeep <= 1 or not 110 <= Tjoin <= Tsurf:
        raise InvalidAtmosphereError('Require 0<=qdeep<=1 and 110<=Tjoin<=Tsurf.')
    cp_d = _as_heat_capacity(cp_d, 'N2')
    cp_v = _as_heat_capacity(cp_v, 'H2O')
    if not np.isfinite(max_step) or max_step <= 0:
        raise InvalidAtmosphereError('max_step must be finite and positive.')
    qsat0 = float(vapor_pressure_limit(Tsurf)) / P0
    if qdeep > qsat0 * (1 + 1.e-9):
        raise InvalidAtmosphereError('qdeep exceeds saturation: it would not be actual surface VMR.')
    xout = np.log(P / P0)
    xmin = min(float(np.min(xout)), 0.)
    y0, ycap = np.log(Tsurf), np.log(Tjoin)
    dry_cp = _HeatCapacity.mix([(1-qdeep, cp_d), (qdeep, cp_v)])
    entropy0 = dry_cp.entropy_log(y0)

    def dry_x(y):
        return (dry_cp.entropy_log(y) - entropy0)/R

    xcap_dry = dry_x(ycap)

    def dry_y_scalar(x):
        if x >= 0:
            return y0
        if x <= xcap_dry:
            return ycap
        return brentq(lambda y: dry_x(y)-x, ycap, y0, xtol=1.e-13)

    def dry_y(x):
        return np.asarray([dry_y_scalar(float(value)) for value in x])

    xlcl = None
    # Only search the part of the dry trajectory inside the water phase-law
    # domain. In particular, never evaluate saturation at the hot bottom.
    ystart = min(y0, np.log(T_CRITICAL))
    yend = dry_y_scalar(xmin)
    if qdeep > 0 and ystart > yend:
        def log_saturation(y):
            # The endpoint uses the analytic critical limit; the public
            # saturation function deliberately excludes the critical point.
            temperature = np.clip(float(np.exp(y)), 110., T_CRITICAL)
            return (np.log(P_CRITICAL) if temperature == T_CRITICAL else
                    float(np.log(water_saturation_pressure(temperature))))

        def dry_saturation(y):
            return log_saturation(y) - np.log(qdeep) - np.log(P0) - dry_x(y)
        start_residual = dry_saturation(ystart)
        if Tsurf >= T_CRITICAL and start_residual < -1.e-9:
            raise InvalidAtmosphereError(
                'Hot adiabat enters the subcritical water domain already '
                'supersaturated; a continuous ideal-gas condensation onset '
                'is unavailable. Increase T_surface or reduce P0/water abundance.')
        if dry_saturation(yend) <= 0:
            ylcl = ystart if start_residual <= 1.e-9 else brentq(dry_saturation, yend, ystart, xtol=1.e-13)
            xlcl = dry_x(ylcl)

    def gradient(x, y, phase=None):
        T = math.exp(y)
        if xlcl is None or x > xlcl + 1.e-12:
            return R / dry_cp.scalar(T)
        # A saturation onset exactly at the critical limit can round a few
        # ulps high when converting from log temperature.
        T = min(T, T_CRITICAL)
        es, L = _scalar_sat_latent(T, phase)
        qsat = 1. if qdeep == 1 else es / (P0 * math.exp(x))
        q = min(qdeep, qsat, 1.)
        ell = L/(R*T)
        cp = (1-q)*cp_d.scalar(T) + q*cp_v.scalar(T)
        return R*((1-q)+ell*q)/((1-q)*cp+R*ell*ell*q)

    reached = Tsurf == Tjoin
    xjoin = 0. if reached else None
    if reached or xmin == 0:
        yout = np.full_like(xout, y0)
    elif xlcl is None or xmin >= xlcl:
        reached = xcap_dry >= xmin
        xjoin = xcap_dry if reached else None
        yout = dry_y(xout)
    elif qdeep == 1:
        # Exact pure-condensable branch: P=es(T) while saturated. Integrating
        # this identity directly avoids false drying caused by ODE drift.
        pcap = float(water_saturation_pressure(Tjoin))
        reached = pcap >= float(np.min(P))
        xjoin = np.log(pcap / P0) if reached else None
        yout = np.empty_like(xout)
        for i, x in enumerate(xout):
            if x > xlcl:
                yout[i] = dry_y_scalar(float(x))
            else:
                if x >= xlcl - 1.e-12:
                    yout[i] = ylcl
                elif P[i] <= pcap:
                    yout[i] = ycap
                else:
                    yout[i] = brentq(lambda y: log_saturation(y) - np.log(P[i]),
                                     ycap, ylcl, xtol=1.e-13)
    elif method == 'fast':
        def rk4(x, y, h):
            # Keep an RK step on one phase branch. Stop exactly at the triple
            # point below; otherwise crossing the latent-heat jump degrades
            # convergence even for very small ordinary pressure steps.
            phase = 'liquid' if y > np.log(T_TRIPLE) + 1.e-13 else 'ice'
            a = gradient(x, y, phase)
            b = gradient(x + h/2, y + h*a/2, phase)
            c = gradient(x + h/2, y + h*b/2, phase)
            d = gradient(x + h, y + h*c, phase)
            return y + h*(a + 2*b + 2*c + d)/6
        xs, ys = [float(xlcl)], [float(ylcl)]
        while xs[-1] > xmin + 1.e-14:
            x, y = xs[-1], ys[-1]
            if y - ycap < 1.e-11:
                reached, xjoin = True, x
                ys[-1] = ycap
                break
            h = max(-max_step, xmin - x)
            # Avoid RK stages outside saturation-law temperature bounds.
            # Cp>R and the water latent factor>1 bound both gradients by 1.
            h = max(h, -0.8 * (y - np.log(110.)))
            yn = rk4(x, y, h)
            yphase = np.log(T_TRIPLE)
            if y > yphase + 1.e-13 and yn < yphase and ycap < yphase:
                h = brentq(lambda hh: rk4(x, y, hh) - yphase, h, 0., xtol=1.e-13)
                yn = yphase
            if yn <= ycap:
                h = brentq(lambda hh: rk4(x, y, hh) - ycap, h, 0., xtol=1.e-13)
                yn = ycap
                reached, xjoin = True, x + h
            xs.append(x + h)
            ys.append(yn)
            if reached:
                break
        spline = PchipInterpolator(np.asarray(xs)[::-1], np.asarray(ys)[::-1])
        yout = spline(np.maximum(xout, xs[-1]))
        yout[xout > xlcl] = dry_y(xout[xout > xlcl])
    elif method == 'reference':
        def stop_at_top(y, x):
            return x[0] - xmin
        stop_at_top.terminal = True
        stop_at_top.direction = -1
        sol = solve_ivp(lambda y, x: [1. / gradient(x[0], y)], (ylcl, ycap), [xlcl],
                        method='DOP853', rtol=2.e-11, atol=2.e-12,
                        dense_output=True, events=stop_at_top, max_step=.015)
        if not sol.success:
            raise RuntimeError(sol.message)
        reached = sol.status == 0
        xend, yend = float(sol.y[0, -1]), float(sol.t[-1])
        xjoin = xend if reached else None
        def invert(x):
            if x <= xend + 1.e-12:
                return yend
            if x >= xlcl - 1.e-12:
                return dry_y_scalar(x)
            return brentq(lambda y: float(sol.sol(y)[0]) - x, yend, ylcl, xtol=1.e-13)
        yout = np.asarray([invert(float(x)) for x in xout])
    else:
        raise InvalidAtmosphereError("method must be 'fast' or 'reference'.")

    T = np.exp(yout)
    Pjoin = P0 * np.exp(xjoin) if reached else None
    if reached:
        T[xout <= xjoin] = Tjoin
    q = np.minimum(qdeep, vapor_pressure_limit(T) / P)
    if reached:
        qjoin = min(qdeep, float(vapor_pressure_limit(Tjoin)) / Pjoin)
        q[xout <= xjoin] = qjoin
    else:
        qjoin = None
    # At pure-steam saturation, q stays exactly one even while condensate forms.
    # Numerical pressure-temperature integration drift must not create dry gas.
    if qdeep == 1:
        q[:] = 1.
        qjoin = 1. if reached else None
    return dict(P=P.copy(), T=T, q=q, Pjoin=Pjoin, Tjoin=Tjoin,
                qjoin=qjoin, reached_join=reached, method=method,
                phase_convention='ice below 273.16 K; liquid at/above 273.16 K',
                thermodynamics='ideal mixture; effective CC latent heat; species-weighted Cp(T) with constant endpoints')


def surface_composition(param):
    """Decode the already sampled composition without choosing a dominant gas.

    Partial-pressure, VMR and CLR sampling all supply these same mole fractions.
    Constant arrays are accepted for compatibility; depleted profiles are not
    valid lower-boundary input. Automatic H2/He splitting matches ExoReL's
    composition preparation and does not mutate the caller's dictionary.
    """
    species = list(param.get('fit_molecules', []))
    if param.get('gas_fill') is not None:
        species.append(param['gas_fill'])
    if not species:
        species = [key[4:] for key in param if key.startswith('vmr_')]
    composition = {}
    for species_name in dict.fromkeys(species):
        try:
            abundance = np.asarray(param['vmr_' + species_name], dtype=float)
        except (KeyError, TypeError, ValueError) as error:
            raise InvalidAtmosphereError('Missing/invalid abundance for ' + species_name) from error
        if abundance.size == 0 or abundance.ndim > 1:
            raise InvalidAtmosphereError('Expected scalar lower-boundary gas abundances.')
        x = float(abundance.flat[0])
        if not np.all(np.isfinite(abundance)) or x < 0 or x > 1:
            raise InvalidAtmosphereError('Gas mole fractions must lie in [0, 1].')
        if not np.allclose(abundance, x, rtol=1e-12, atol=0):
            raise InvalidAtmosphereError('Temperature requires undepleted lower-boundary abundances.')
        composition[species_name] = x
    total = sum(composition.values())
    if not np.isfinite(total) or not np.isclose(total, 1., rtol=0, atol=1e-10):
        raise InvalidAtmosphereError('Lower-boundary gas mole fractions must sum to one.')
    # Remove only floating-point summation error, not physically invalid inputs.
    composition = {species_name: x / total for species_name, x in composition.items()}
    automatic_h2_he = (not param.get('rocky', True)
                       and param.get('gas_fill') == 'H2'
                       and 'He' not in param.get('fit_molecules', [])
                       and param.get('H2_He_ratio', 0.) > 0)
    if automatic_h2_he:
        fraction = float(param['H2_He_ratio'])
        if not 0 < fraction <= 1:
            raise InvalidAtmosphereError('H2_He_ratio must lie in (0, 1].')
        background = composition['H2']
        composition['H2'] = background * fraction
        composition['He'] = background * (1 - fraction)
    return composition


def composition_heat_capacity_models(param):
    """Return bottom water VMR and dry/water Cp(T) models, constructed once."""
    composition = surface_composition(param)
    overrides = param.get('molar_heat_capacities') or {}
    if not isinstance(overrides, dict):
        raise InvalidAtmosphereError('molar_heat_capacities must be a species-to-Cp dictionary.')
    capacities = {}
    needed = {name for name, abundance in composition.items() if abundance > 0} | {'H2O'}
    for name in needed:
        capacities[name] = (_HeatCapacity.constant(overrides[name]) if name in overrides
                            else _species_heat_capacity(name))
    water = composition.get('H2O', 0.)
    dry_total = sum(x for name, x in composition.items() if name != 'H2O')
    # In pure steam the dry contribution is identically zero. Reuse the
    # water model as an unused placeholder instead of inventing a dry Cp.
    dry_cp = (_HeatCapacity.mix([(x/dry_total, capacities[name])
                                for name, x in composition.items() if name != 'H2O' and x > 0])
              if dry_total > 0 else capacities['H2O'])
    return water, dry_cp, capacities['H2O']


def composition_heat_capacities(param, T=298.15):
    """Evaluate composition-weighted molar heat capacities at scalar/array T."""
    water, dry_cp, vapor_cp = composition_heat_capacity_models(param)
    return water, dry_cp(T), vapor_cp(T)


def irradiation_basis(tau, gamma):
    """Guillot Eq.49 irradiation shape, stable at small optical depth."""
    tau = np.asarray(tau, dtype=float)
    u = gamma * tau
    return (1 + (-np.expm1(-u) + .5*u*np.exp(-u))/gamma
            + gamma*(1-.5*tau**2)*expn(2, u))


def radiative_temperature(P, T_join, P_join, tau_join, gamma, eta):
    """Normalized upper branch; eta is a T^4 weight at the join, not flux."""
    P = np.asarray(P, dtype=float)
    if tau_join == 0:
        return np.full_like(P, T_join)
    tau = tau_join * P / P_join
    shape = (eta*(2/3+tau)/(2/3+tau_join)
             + (1-eta)*irradiation_basis(tau, gamma)/irradiation_basis(tau_join, gamma))
    if np.any(shape <= 0) or not np.all(np.isfinite(shape)):
        raise InvalidAtmosphereError('Unresolved or invalid radiative temperature shape.')
    return T_join * shape**.25


def coupled_temperature_profile(param):
    """Evaluate the six-parameter profile on param['P'] in the original order.

    Five thermal parameters: T_surface, T_join, tau_join, gamma, eta.
    T_surface anchors the bottom at P0; no physical solid surface is required.
    The sixth quantity is the existing decoded vmr_H2O. There is no second
    water parameter. Other gas abundances determine the dry-mixture heat
    capacity. Cloud-opacity flags do not affect the temperature profile.
    """
    required = ('T_surface', 'T_join', 'tau_join', 'gamma', 'eta', 'P0', 'P')
    missing = [name for name in required if param.get(name) is None]
    if missing:
        raise InvalidAtmosphereError('Missing coupled-profile inputs: ' + ', '.join(missing))
    try:
        Ts, Tj, tauj, gamma, eta, P0 = (float(param[name]) for name in required[:-1])
        P = np.asarray(param['P'], dtype=float)
    except (TypeError, ValueError) as error:
        raise InvalidAtmosphereError('Thermal parameters must be finite scalars.') from error
    if not np.all(np.isfinite([Ts, Tj, tauj, gamma, eta, P0])):
        raise InvalidAtmosphereError('Thermal parameters must be finite scalars.')
    if not 110 <= Tj <= Ts:
        raise InvalidAtmosphereError('Require 110 <= T_join <= T_surface.')
    if tauj < 0 or gamma <= 0 or not 0 <= eta <= 1 or P0 <= 0:
        raise InvalidAtmosphereError('Require tau_join>=0, gamma>0, 0<=eta<=1, P0>0.')
    if P.ndim != 1 or not P.size or np.any(~np.isfinite(P)) or np.any(P <= 0) or np.any(P > P0*(1+1e-12)):
        raise InvalidAtmosphereError('Pressure must be a positive 1-D grid at or above the surface.')
    q, cp_d, cp_v = composition_heat_capacity_models(param)
    maximum = min(1., float(vapor_pressure_limit(Ts))/P0)
    if q > maximum * (1+1e-10):
        raise InvalidAtmosphereError('Surface H2O vapor exceeds saturation.')
    if Tj == Ts:
        T = radiative_temperature(P, Tj, P0, tauj, gamma, eta)
    else:
        convective = convection_profile(P, Ts, q, Tj, P0=P0, cp_d=cp_d, cp_v=cp_v)
        T = convective['T']
        Pj = convective['Pjoin']
        if Pj is not None:
            upper = P <= Pj
            T[upper] = radiative_temperature(P[upper], Tj, Pj, tauj, gamma, eta)
    # Radiative inversions may leave the domain even when boundary inputs do not.
    _valid_atmospheric_temperature(T)
    if q == 1 and np.any(vapor_pressure_limit(T)/P < 1-1e-8):
        raise InvalidAtmosphereError('The radiative branch would supersaturate pure water gas.')
    return T
