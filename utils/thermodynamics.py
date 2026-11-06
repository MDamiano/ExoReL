"""Saturation pressures and coupled parametric temperature profiles.

All pressures are total gas pressures in Pa, temperatures K, and abundances
molar/volume fractions. This is a temperature-continuous retrieval model, not
a radiative-convective equilibrium or buoyancy/stability solver. Only H2O
latent heating enters the thermal gradient; other species form a dry mixture.

The constant species molar Cp approximation uses ideal-gas reference values
at 298.15 K (J mol^-1 K^-1), weighted for EACH sampled composition. It does
not assume a fixed dominant/filling gas or claim temperature-dependent Cp.
Overrides may be supplied as param['molar_heat_capacities'] (same units).
NIST WebBook SRD69, Chase (1998) low-temperature Shomate fits evaluated at
298.15 K: https://webbook.nist.gov/cgi/cbook.cgi?ID=C<CAS digits>&Mask=1
N2 7727379; O2 7782447; CO2 124389; CO 630080; H2 1333740;
NH3 7664417; H2S 7783064; SO2 7446095; N2O 10024972; O3 10028156.
CH4: WebBook 74828, Gurvich et al. (1989) tabulated Cp at 298.15 K.
H2O: https://janaf.nist.gov/tables/H-064.html (33.590 at 298.15 K).
He uses the monatomic ideal-gas value 5R/2.

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
CP_D = 29.12379
CP_V = 33.590
_A = np.array([-7.85951783, 1.84408259, -11.7866497, 22.6807411,
               -15.9618719, 1.80122502])
_B = np.array([1., 1.5, 3., 3.5, 4., 7.5])
_COEFFICIENTS = tuple(zip(_A.tolist(), _B.tolist()))


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


def latent_molar(T):
    """Effective L=R*T**2*d(log(es))/dT; ideal-CC consistent, not real-EOS L."""
    T = _valid_temperature(T)
    theta = 1.0 - T / T_CRITICAL
    S = np.sum(_A * theta[..., None] ** _B, axis=-1)
    dS = np.sum(_A * _B * theta[..., None] ** (_B - 1), axis=-1)
    liquid = -T_CRITICAL / T**2 * S - dS / T
    ice = 5723.265 / T**2 + 3.53068 / T - .00728332
    return R * T**2 * np.where(T < T_TRIPLE, ice, liquid)


def dry_gradient(q, cp_d=CP_D, cp_v=CP_V):
    """d ln(T)/d ln(P), unsaturated fixed-composition ideal gas."""
    q = np.asarray(q, dtype=float)
    if np.any((q < 0) | (q > 1)):
        raise InvalidAtmosphereError('Water VMR must be in [0, 1].')
    return R / ((1 - q) * cp_d + q * cp_v)


def moist_gradient(T, q, cp_d=CP_D, cp_v=CP_V):
    """Nondilute saturated pseudoadiabatic gradient, regular at q=0 and q=1."""
    T = np.asarray(T, dtype=float)
    q = np.asarray(q, dtype=float)
    if np.any((q < 0) | (q > 1)):
        raise InvalidAtmosphereError('Water VMR must be in [0, 1].')
    ell = latent_molar(T) / (R * T)
    cp = (1 - q) * cp_d + q * cp_v
    return R * ((1 - q) + ell * q) / ((1 - q) * cp + R * ell**2 * q)


def convection_profile(P, Tsurf, qdeep, Tjoin=200., P0=1.e5, method='fast',
                       max_step=0.02, cp_d=CP_D, cp_v=CP_V):
    """Surface-anchored dry/moist profile with an isothermal cap.

    P is a 1D array in either order, with 0<P<=P0. Output order matches input.
    Pjoin is None if the cap is not reached by min(P). qdeep is the actual
    surface water VMR and must not exceed surface saturation. 'fast' uses RK4
    in log pressure; 'reference' integrates log pressure versus log temperature
    using adaptive DOP853 and root-inverts the independent-variable solution.
    Both use the same thermodynamic assumptions but different integration paths.
    """
    P = np.asarray(P, dtype=float)
    if method not in ('fast', 'reference'):
        raise InvalidAtmosphereError("method must be 'fast' or 'reference'.")
    Tsurf, Tjoin, qdeep, P0 = map(float, (Tsurf, Tjoin, qdeep, P0))
    _valid_temperature([Tsurf, Tjoin])
    if P.ndim != 1 or not len(P) or np.any(~np.isfinite(P)) or np.any(P <= 0):
        raise InvalidAtmosphereError('P must be a nonempty finite positive 1D array.')
    if not np.isfinite(P0) or P0 <= 0 or np.any(P > P0 * (1 + 1.e-12)):
        raise InvalidAtmosphereError('All pressure points must lie at or above the surface.')
    if not 0 <= qdeep <= 1 or not 110 <= Tjoin <= Tsurf:
        raise InvalidAtmosphereError('Require 0<=qdeep<=1 and 110<=Tjoin<=Tsurf.')
    if cp_d <= 0 or cp_v <= 0 or max_step <= 0:
        raise InvalidAtmosphereError('Heat capacities and max_step must be positive.')
    qsat0 = float(water_saturation_pressure(Tsurf)) / P0
    if qdeep > qsat0 * (1 + 1.e-9):
        raise InvalidAtmosphereError('qdeep exceeds saturation: it would not be actual surface VMR.')
    xout = np.log(P / P0)
    xmin = min(float(np.min(xout)), 0.)
    y0, ycap = np.log(Tsurf), np.log(Tjoin)
    gdry = float(dry_gradient(qdeep, cp_d, cp_v))
    xcap_dry = (ycap - y0) / gdry
    xlcl = None
    if qdeep > 0 and Tsurf > Tjoin:
        def dry_saturation(y):
            return float(np.log(water_saturation_pressure(np.exp(y)))) - np.log(qdeep * P0) - (y-y0)/gdry
        if dry_saturation(ycap) <= 0:
            ylcl = y0 if dry_saturation(y0) <= 1.e-9 else brentq(dry_saturation, ycap, y0, xtol=1.e-13)
            xlcl = (ylcl - y0) / gdry

    def gradient(x, y, phase=None):
        T = math.exp(y)
        if xlcl is None or x > xlcl + 1.e-12:
            return gdry
        es, L = _scalar_sat_latent(T, phase)
        qsat = 1. if qdeep == 1 else es / (P0 * math.exp(x))
        q = min(qdeep, qsat, 1.)
        ell = L/(R*T)
        cp = (1-q)*cp_d + q*cp_v
        return R*((1-q)+ell*q)/((1-q)*cp+R*ell*ell*q)

    reached = Tsurf == Tjoin
    xjoin = 0. if reached else None
    if reached or xmin == 0:
        yout = np.full_like(xout, y0)
    elif xlcl is None or xmin >= xlcl:
        reached = xcap_dry >= xmin
        xjoin = xcap_dry if reached else None
        yout = np.maximum(ycap, y0 + gdry * xout)
    elif qdeep == 1:
        # Exact pure-condensable branch: P=es(T) while saturated. Integrating
        # this identity directly avoids false drying caused by ODE drift.
        pcap = float(water_saturation_pressure(Tjoin))
        reached = pcap >= float(np.min(P))
        xjoin = np.log(pcap / P0) if reached else None
        yout = y0 + gdry * xout
        for i, x in enumerate(xout):
            if x <= xlcl:
                if x >= xlcl - 1.e-12:
                    yout[i] = ylcl
                elif P[i] <= pcap:
                    yout[i] = ycap
                else:
                    yout[i] = brentq(lambda y: float(np.log(water_saturation_pressure(np.exp(y)))) - np.log(P[i]),
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
            h = max(h, -0.8 * (y - np.log(110.)) / gdry)
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
        yout[xout > xlcl] = y0 + gdry*xout[xout > xlcl]
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
                return y0 + gdry*x
            return brentq(lambda y: float(sol.sol(y)[0]) - x, yend, ylcl, xtol=1.e-13)
        yout = np.asarray([invert(float(x)) for x in xout])
    else:
        raise InvalidAtmosphereError("method must be 'fast' or 'reference'.")

    T = np.exp(yout)
    Pjoin = P0 * np.exp(xjoin) if reached else None
    if reached:
        T[xout <= xjoin] = Tjoin
    q = np.minimum(qdeep, water_saturation_pressure(T) / P)
    if reached:
        qjoin = min(qdeep, float(water_saturation_pressure(Tjoin)) / Pjoin)
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
                thermodynamics='ideal mixture; effective CC latent heat; species-weighted constant molar cp')


MOLAR_HEAT_CAPACITIES = {
    'H2O': 33.590, 'NH3': 35.64710, 'CH4': 35.69, 'H2S': 34.19732,
    'SO2': 39.87358, 'CO2': 37.12996, 'CO': 29.14870, 'O2': 29.38263,
    'O3': 39.23006, 'N2O': 38.61099, 'N2': 29.12379, 'He': 2.5 * R,
    'H2': 28.83730,
}


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


def composition_heat_capacities(param):
    """Return water VMR, dry-mixture Cp, water Cp (molar SI units)."""
    composition = surface_composition(param)
    capacities = dict(MOLAR_HEAT_CAPACITIES)
    overrides = param.get('molar_heat_capacities') or {}
    if not isinstance(overrides, dict):
        raise InvalidAtmosphereError('molar_heat_capacities must be a species-to-Cp dictionary.')
    capacities.update(overrides)
    needed = {name for name, abundance in composition.items() if abundance > 0} | {'H2O'}
    for name in needed:
        try:
            cp = float(capacities[name])
        except (KeyError, TypeError, ValueError) as error:
            raise InvalidAtmosphereError('Supply molar heat capacity for ' + name) from error
        if not np.isfinite(cp) or cp <= R:
            raise InvalidAtmosphereError('Gas molar Cp must be finite and greater than R.')
        capacities[name] = cp
    water = composition.get('H2O', 0.)
    dry_total = sum(x for name, x in composition.items() if name != 'H2O')
    dry_cp = (sum(x * capacities[name] for name, x in composition.items()
                  if name != 'H2O' and x > 0) / dry_total) if dry_total > 0 else CP_D
    return water, dry_cp, capacities['H2O']


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
    if not 110 <= Tj <= Ts < T_CRITICAL:
        raise InvalidAtmosphereError('Require 110 <= T_join <= T_surface < 647.096 K.')
    if tauj < 0 or gamma <= 0 or not 0 <= eta <= 1 or P0 <= 0:
        raise InvalidAtmosphereError('Require tau_join>=0, gamma>0, 0<=eta<=1, P0>0.')
    if P.ndim != 1 or not P.size or np.any(~np.isfinite(P)) or np.any(P <= 0) or np.any(P > P0*(1+1e-12)):
        raise InvalidAtmosphereError('Pressure must be a positive 1-D grid at or above the surface.')
    q, cp_d, cp_v = composition_heat_capacities(param)
    maximum = min(1., float(water_saturation_pressure(Ts))/P0)
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
    _valid_temperature(T)
    if q == 1 and np.any(water_saturation_pressure(T)/P < 1-1e-8):
        raise InvalidAtmosphereError('The radiative branch would supersaturate pure water gas.')
    return T
