import os
import sys
import copy
import math
import json
import numpy as np
import pymultinest
import matplotlib.pyplot as plt
from matplotlib import ticker
from astropy import constants as const
from skbio.stats.composition import clr, clr_inv

from .__utils import find_nearest, model_finalizzation
from .__forward import FORWARD_MODEL, FORWARD_DATASET, FORWARD_AI


def _instantiate_forward_model(param):
    model_type = param.get('physics_model')
    if model_type == 'radiative_transfer':
        return FORWARD_MODEL(param, retrieval=False, canc_metadata=True)
    if model_type == 'dataset':
        return FORWARD_DATASET(param, dataset_dir=param['dataset_dir'])
    if model_type == 'AI_model':
        return FORWARD_AI(param)
    raise ValueError('Unknown physics_model: ' + str(model_type))


def plot_nest_spec(mnest, cube, solutions=None):
    """Plot retrieved spectrum and confidence bands.

    Parameters
    - mnest: MULTINEST instance (provides `param`, `cube_to_param`, `adjust_for_cld_frac`).
    - cube: parameter cube (array-like)
    - solutions: Optional solution index for filenames

    Notes
    - Preserves all original behaviors/outputs while improving visuals and avoiding
      repeated I/O where possible.
    """
    # Lightweight aesthetic improvements
    try:
        plt.style.use('seaborn-v0_8-white')
    except Exception:
        pass

    mnest.cube_to_param(cube)

    # Helper: load target R=500 wavelength bins once
    def _load_target_bins(param):
        if param['mol_custom_wl']:
            new_wl = np.loadtxt(param['pkg_dir'] + 'forward_mod/Data/wl_bins/bins_02_50_R500.dat')
        else:
            new_wl = np.loadtxt(param['pkg_dir'] + 'forward_mod/Data/wl_bins/bins_02_20_R500.dat')
        return new_wl, np.mean(new_wl, axis=1)

    # Single observation
    if mnest.param['obs_numb'] is None:
        fig = plt.figure(figsize=(8.5, 5.0), dpi=130)

        # Data
        plt.errorbar(
            mnest.param['spectrum']['wl'],
            mnest.param['spectrum']['Fplanet'],
            yerr=mnest.param['spectrum']['error_p'],
            linestyle='', linewidth=0.8, color='#222222', marker='o',
            markerfacecolor='#e24a33', markeredgecolor='#222222', markersize=4.5,
            capsize=2.0, label='Data')

        # Model on native grid
        mod = _instantiate_forward_model(mnest.param)
        alb_wl, alb = mod.run_forward()
        if mnest.param['fit_wtr_cld'] and mnest.param['cld_frac'] != 1.0:
            alb = mnest.adjust_for_cld_frac(alb, cube)
            mnest.cube_to_param(cube)
        _, model_native = model_finalizzation(mnest.param, alb_wl, alb,
                                              planet_albedo=mnest.param['albedo_calc'],
                                              fp_over_fs=mnest.param['fp_over_fs'])
        plt.plot(
            mnest.param['spectrum']['wl'], model_native, linestyle='', color='#1f77b4',
            marker='D', markerfacecolor='#4c78a8', markeredgecolor='#1f77b4', markersize=4.0,
            label='MAP (data native)')

        # Prepare R=500 grid once
        new_wl, new_wl_central = _load_target_bins(mnest.param)
        if mnest.param['mol_custom_wl']:
            start, stop = 0, len(new_wl_central) - 1
        else:
            start = find_nearest(new_wl_central, min(mnest.param['spectrum']['wl']) - 0.05)
            stop = find_nearest(new_wl_central, max(mnest.param['spectrum']['wl']) + 0.05)

        # Temporarily swap spectrum grid to R=500 for plotting a smooth curve
        if mnest.param['spectrum']['bins']:
            temp_spec = np.array([mnest.param['spectrum']['wl_low'],
                                  mnest.param['spectrum']['wl_high'],
                                  mnest.param['spectrum']['wl']]).T
        else:
            temp_spec = mnest.param['spectrum']['wl'] + 0.0
        mnest.param['spectrum']['wl'] = new_wl_central[start:stop]
        mnest.param['spectrum']['wl_low'] = new_wl[start:stop, 0]
        mnest.param['spectrum']['wl_high'] = new_wl[start:stop, 1]

        temp_min, temp_max = mnest.param['min_wl'] + 0.0, mnest.param['max_wl'] + 0.0
        mnest.param['min_wl'] = float(np.min(mnest.param['spectrum']['wl']))
        mnest.param['max_wl'] = float(np.max(mnest.param['spectrum']['wl']))
        mnest.param['start_c_wl_grid'] = find_nearest(mnest.param['wl_C_grid'], mnest.param['min_wl']) - 35
        mnest.param['stop_c_wl_grid'] = find_nearest(mnest.param['wl_C_grid'], mnest.param['max_wl']) + 35

        # Model on R=500 grid
        mod = _instantiate_forward_model(mnest.param)
        alb_wl, alb = mod.run_forward()
        if mnest.param['fit_wtr_cld'] and mnest.param['cld_frac'] != 1.0:
            alb = mnest.adjust_for_cld_frac(alb, cube)
            mnest.cube_to_param(cube)
        wl, model = model_finalizzation(mnest.param, alb_wl, alb,
                                        planet_albedo=mnest.param['albedo_calc'],
                                        fp_over_fs=mnest.param['fp_over_fs'])
        plt.plot(wl, model, color='#404784', linewidth=1.2, label='MAP (R=500)')

        best_fit = np.array([wl, model]).T

        # Optional credible intervals from random samples
        rs_path = mnest.param['out_dir'] + 'random_samples.dat'
        if os.path.isfile(rs_path):
            fl = np.loadtxt(rs_path)
            q50 = np.quantile(fl[:, 1:], 0.5, axis=1)
            q16, q84 = np.quantile(fl[:, 1:], [0.16, 0.84], axis=1)
            q2, q98 = np.quantile(fl[:, 1:], [0.0225, 0.9775], axis=1)
            q003, q997 = np.quantile(fl[:, 1:], [0.00135, 0.99865], axis=1)

            bands = [q16, q84, q2, q98, q003, q997]
            for qq in bands:
                qq[qq < 0.0] = 0.0

            p16, p84 = best_fit[:, 1] + (q16 - q50), best_fit[:, 1] + (q84 - q50)
            p2, p98 = best_fit[:, 1] + (q2 - q50), best_fit[:, 1] + (q98 - q50)
            p003, p997 = best_fit[:, 1] + (q003 - q50), best_fit[:, 1] + (q997 - q50)

            best_fit = np.column_stack([best_fit, p84, p16, p98, p2, p997, p003])

            plt.fill_between(fl[:, 0], p003, p997, ec=(0, 0, 0, 0), fc=(64/255, 71/255, 132/255, 0.20), label='3σ')
            plt.fill_between(fl[:, 0], p2, p98, ec=(0, 0, 0, 0), fc=(64/255, 71/255, 132/255, 0.35), label='2σ')
            plt.fill_between(fl[:, 0], p16, p84, ec=(0, 0, 0, 0), fc=(64/255, 71/255, 132/255, 0.50), label='1σ')

        # Save best fit table
        if solutions is None:
            np.savetxt(mnest.param['out_dir'] + 'Best_fit.dat', best_fit)
        else:
            np.savetxt(mnest.param['out_dir'] + f'Best_fit_{solutions}.dat', best_fit)

        # Restore spectrum grid and bounds
        if mnest.param['spectrum']['bins']:
            mnest.param['spectrum']['wl'] = temp_spec[:, 2]
            mnest.param['spectrum']['wl_low'] = temp_spec[:, 0]
            mnest.param['spectrum']['wl_high'] = temp_spec[:, 1]
        else:
            mnest.param['spectrum']['wl'] = temp_spec + 0.0
        mnest.param['min_wl'] = copy.deepcopy(temp_min)
        mnest.param['max_wl'] = copy.deepcopy(temp_max)
        mnest.param['start_c_wl_grid'] = find_nearest(mnest.param['wl_C_grid'], mnest.param['min_wl']) - 35
        mnest.param['stop_c_wl_grid'] = find_nearest(mnest.param['wl_C_grid'], mnest.param['max_wl']) + 35

        # Labels and cosmetics
        plt.legend(frameon=False)
        plt.xlabel('Wavelength [$\mu$m]')
        if mnest.param['albedo_calc']:
            plt.ylabel('Albedo')
        elif mnest.param['fp_over_fs']:
            plt.ylabel('Contrast Ratio (F$_p$/F$_{\star}$)')
        else:
            plt.ylabel('Planetary flux [W/m$^2$]')
        fig.tight_layout()

    # Multiple observations stacked
    else:
        fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0}, figsize=(8.5, 6.0), dpi=130)
        new_wl, new_wl_central = _load_target_bins(mnest.param)
        mnest.param['spectrum']['bins'] = False

        for obs in range(0, mnest.param['obs_numb']):
            mnest.cube_to_param(cube, n_obs=obs)
            mod = _instantiate_forward_model(mnest.param)
            alb_wl, alb = mod.run_forward()
            if mnest.param['fit_cld_frac'] and mnest.param['fit_wtr_cld'] and mnest.param['cld_frac'] != 1.0:
                alb = mnest.adjust_for_cld_frac(alb, cube)
                mnest.cube_to_param(cube, n_obs=obs)
            _, model_native = model_finalizzation(mnest.param, alb_wl, alb,
                                                  planet_albedo=mnest.param['albedo_calc'],
                                                  fp_over_fs=mnest.param['fp_over_fs'], n_obs=obs)

            axs[obs].plot(mnest.param['spectrum'][str(obs)]['wl'], model_native, linestyle='', color='#1f77b4',
                          marker='D', markerfacecolor='#4c78a8', markeredgecolor='#1f77b4', markersize=4.0)

            if mnest.param['fit_phi']:
                lab = f'Input spectrum {obs}'
            else:
                lab = 'Data, $\\phi=$' + str(mnest.param['phi' + str(obs)] * 180.0 / math.pi) + ' deg'
            axs[obs].errorbar(
                mnest.param['spectrum'][str(obs)]['wl'],
                mnest.param['spectrum'][str(obs)]['Fplanet'],
                yerr=mnest.param['spectrum'][str(obs)]['error_p'],
                linestyle='', linewidth=0.8, color='#222222', marker='o', markerfacecolor='#e24a33',
                markeredgecolor='#222222', markersize=4.5, capsize=2.0, label=lab)

            start = find_nearest(new_wl_central, min(mnest.param['spectrum'][str(obs)]['wl']) - 0.05)
            stop = find_nearest(new_wl_central, max(mnest.param['spectrum'][str(obs)]['wl']) + 0.05)

            temp = mnest.param['spectrum'][str(obs)]['wl'] + 0.0
            mnest.param['spectrum'][str(obs)]['wl'] = new_wl_central[start:stop]

            mod = _instantiate_forward_model(mnest.param)
            alb_wl, alb = mod.run_forward()
            alb_wl *= 10.0 ** (-3.0)
            if mnest.param['fit_cld_frac'] and mnest.param['fit_wtr_cld'] and mnest.param['cld_frac'] != 1.0:
                alb = mnest.adjust_for_cld_frac(alb, cube)
                mnest.cube_to_param(cube, n_obs=obs)
            wl, model = model_finalizzation(mnest.param, alb_wl, alb,
                                            planet_albedo=mnest.param['albedo_calc'],
                                            fp_over_fs=mnest.param['fp_over_fs'], n_obs=obs)

            axs[obs].plot(wl, model, color='#404784', linewidth=1.2, label='MAP (R=500)')
            mnest.param['spectrum'][str(obs)]['wl'] = temp + 0.0

            axs[obs].legend(frameon=False)
            axs[obs].set_ylim([-0.2 * min(model), max(model) + 0.2 * max(model)])

        if mnest.param['albedo_calc']:
            fig.text(0.04, 0.5, 'Albedo', va='center', rotation='vertical')
        else:
            if mnest.param['fp_over_fs']:
                fig.text(0.04, 0.5, 'F$_p$/F$_{\\star}$', va='center', rotation='vertical')
            else:
                fig.text(0.04, 0.5, 'Planetary flux [W/m$^2$]', va='center', rotation='vertical')
        fig.text(0.5, 0.04, 'Wavelength [$\\mu$m]', ha='center')

    # Save figure
    if solutions is None:
        plt.savefig(mnest.param['out_dir'] + 'Nest_spectrum.pdf')
    else:
        plt.savefig(mnest.param['out_dir'] + f'Nest_spectrum (solution {solutions}).pdf')
    plt.close()


def plot_posteriors_old_not_used(self, prefix, multinest_results, parameters, mds_orig):
        from numpy import log
        from six.moves import range # type: ignore
        import logging
        import types
        from matplotlib.ticker import MaxNLocator, NullLocator
        from matplotlib.colors import LinearSegmentedColormap, colorConverter
        from matplotlib.ticker import ScalarFormatter
        from scipy.ndimage import gaussian_filter as norm_kde
        from scipy.stats import gaussian_kde

        try:
            str_type = types.StringTypes
            float_type = types.FloatType
            int_type = types.IntType
        except:
            str_type = str
            float_type = float
            int_type = int

        # Light visual polish
        try:
            plt.style.use('seaborn-v0_8-white')
        except Exception:
            pass

        SQRTEPS = math.sqrt(float(np.finfo(np.float64).eps))

        def _posteriors_gas_to_vmr(prefix, modes=None):
            if modes is None:
                os.system('cp ' + prefix + '.txt ' + prefix + 'original.txt')
                a = np.loadtxt(prefix + '.txt')
            else:
                os.system('cp ' + prefix + 'solution' + str(modes) + '.txt ' + prefix + 'solution' + str(modes) + '_original.txt')
                a = np.loadtxt(prefix + 'solution' + str(modes) + '.txt')

            b = np.ones((len(a[:, 0]), len(a[0, :]) + 2))

            if self.param['fit_p0'] and self.param['gas_par_space'] != 'partial_pressure':
                b[:, 0:6] = a[:, 0:6] + 0.0
                z = 6
            elif self.param['gas_par_space'] == 'partial_pressure':
                b[:, 0:2] = a[:, 0:2] + 0.0
                b[:, 3:6] = a[:, 2:5] + 0.0
                z = 5
            else:
                b[:, 0:5] = a[:, 0:5] + 0.0
                z = 8

            if not self.param['fit_wtr_cld']:
                z -= 3

            if self.param['fit_amm_cld']:
                b[:, 0:z + 3] = a[:, 0:z + 3] + 0.0
                z += 3

            volume_mixing_ratio = {}
            if self.param['gas_par_space'] == 'centered_log_ratio' or self.param['gas_par_space'] == 'clr':
                c_l_r = np.array(a[:, z:z + len(self.param['fit_molecules'])])
                c_l_r = np.concatenate((c_l_r, np.array([-np.sum(c_l_r, axis=1)]).T), axis=1)
                v_m_r = clr_inv(c_l_r)
                for i, mol in enumerate(self.param['fit_molecules']):
                    volume_mixing_ratio[mol] = v_m_r[:, i]
                volume_mixing_ratio[self.param['gas_fill']] = v_m_r[:, -1]
            elif self.param['gas_par_space'] == 'volume_mixing_ratio' or self.param['gas_par_space'] == 'vmr':
                volume_mixing_ratio[self.param['gas_fill']] = np.ones(len(a[:, 0]))
                for i, mol in enumerate(self.param['fit_molecules']):
                    volume_mixing_ratio[mol] = 10.0 ** np.array(a[:, z + i])
                    volume_mixing_ratio[self.param['gas_fill']] -= 10.0 ** np.array(a[:, z + i])
            elif self.param['gas_par_space'] == 'partial_pressure':
                b[:, 2] = np.sum(10.0 ** np.array(a[:, z:z + len(self.param['fit_molecules'])]), axis=1)
                for i, mol in enumerate(self.param['fit_molecules']):
                    volume_mixing_ratio[mol] = (10.0 ** np.array(a[:, z + i])) / b[:, 2]
                b[:, 2] = np.log10(b[:, 2])

            mmm = np.zeros(len(a[:, 0]))
            for mol in volume_mixing_ratio.keys():
                mmm += volume_mixing_ratio[mol] * self.param['mm'][mol]

            if self.param['gas_par_space'] != 'partial_pressure':
                for i, mol in enumerate(self.param['fit_molecules']):
                    b[:, z + i] = np.log10(volume_mixing_ratio[mol])
                if self.param['gas_fill'] is not None:
                    b[:, z + i + 1] = np.log10(volume_mixing_ratio[self.param['gas_fill']])
            else:
                for i, mol in enumerate(self.param['fit_molecules']):
                    b[:, (z + 1) + i] = np.log10(volume_mixing_ratio[mol])

            b[:, z + i + 2:-1] = a[:, z + i + 1:] + 0.0

            if self.param['fit_p_size']:
                locate_mp_rp = 4
            else:
                locate_mp_rp = 3

            if self.param['rocky'] and self.param['fit_Mp']:
                b[:, -locate_mp_rp] *= (const.M_jup.value / const.M_earth.value)
            if self.param['rocky'] and self.param['fit_Rp']:
                b[:, -(locate_mp_rp - 1)] *= (const.R_jup.value / const.R_earth.value)

            b[:, -1] = np.array(mmm) + 0.0

            if modes is None:
                np.savetxt(prefix + '.txt', b)
            else:
                np.savetxt(prefix + 'solution' + str(modes) + '.txt', b)

        def _corner_parameters():
            if os.path.isfile(prefix + 'params_original.json'):
                pass
            else:
                os.system('mv ' + prefix + 'params.json ' + prefix + 'params_original.json')

            par = []
            if self.param['fit_p0'] and self.param['gas_par_space'] != 'partial_pressure':
                par.append("Log(P$_0$ [Pa])")
            elif not self.param['fit_p0'] and self.param['gas_par_space'] == 'partial_pressure':
                par.append("Log(P$_0$ [Pa]) (derived)")
            if self.param['fit_wtr_cld']:
                par.append("Log(P$_{top, H_2O}$ [Pa])")
                par.append("Log(D$_{H_2O}$ [Pa])")
                par.append("Log(CR$_{H_2O}$)")
            if self.param['fit_amm_cld']:
                par.append("Log(P$_{top, NH_3}$ [Pa])")
                par.append("Log(D$_{NH_3}$ [Pa])")
                par.append("Log(CR$_{NH_3}$)")
            for mol in self.param['fit_molecules']:
                par.append(self.param['formatted_labels'][mol])
                if self.param['gas_fill'] is not None:
                    if self.param['rocky']:
                        par.append(self.param['formatted_labels'][self.param['gas_fill']] + " (derived)")
                    else:
                        par.append("Log(H$_2$ + He) (derived)")
            if self.param['fit_ag']:
                if self.param['surface_albedo_parameters'] == int(1):
                    par.append("$a_{surf}$")
                elif self.param['surface_albedo_parameters'] == int(3):
                    par.append("$a_{surf, 1}$")
                    par.append("$a_{surf, 2}$")
                    par.append("$\lambda_{surf, 1}$ [$\mu$m]")
                elif self.param['surface_albedo_parameters'] == int(5):
                    par.append("$a_{surf, 1}$")
                    par.append("$a_{surf, 2}$")
                    par.append("$a_{surf, 3}$")
                    par.append("$\lambda_{surf, 1}$ [$\mu$m]")
                    par.append("$\lambda_{surf, 2}$ [$\mu$m]")
            if self.param['fit_cld_frac']:
                par.append("Log(cld frac)")
            if self.param['fit_g']:
                par.append("Log(g [m/s$^2$])")
            if self.param['fit_Mp']:
                if self.param['rocky']:
                    par.append("M$_p$ [M$_\oplus$]")
                else:
                    par.append("M$_p$ [M$_J$]")
            if self.param['fit_Rp']:
                if self.param['rocky']:
                    par.append("R$_p$ [R$_\oplus$]")
                else:
                    par.append("R$_p$ [R$_{Jup}$]")
            if self.param['fit_p_size'] and self.param['p_size_type'] == 'constant':
                par.append("Log(P$_{size}$ [$\mu$m])")
            elif self.param['fit_p_size'] and self.param['p_size_type'] == 'factor':
                par.append("Log(P$_{size, fctr})$")
            par.append("$\mu$ (derived)")
            json.dump(par, open(prefix + 'params.json', 'w'))

        def _quantile(x, q, weights=None):
            """
            Compute (weighted) quantiles from an input set of samples.
            Parameters
            ----------
            x : `~numpy.ndarray` with shape (nsamps,)
                Input samples.
            q : `~numpy.ndarray` with shape (nquantiles,)
               The list of quantiles to compute from `[0., 1.]`.
            weights : `~numpy.ndarray` with shape (nsamps,), optional
                The associated weight from each sample.
            Returns
            -------
            quantiles : `~numpy.ndarray` with shape (nquantiles,)
                The weighted sample quantiles computed at `q`.
            """

            # Initial check.
            x = np.atleast_1d(x)
            q = np.atleast_1d(q)

            # Quantile check.
            if np.any(q < 0.0) or np.any(q > 1.0):
                raise ValueError("Quantiles must be between 0. and 1.")

            if weights is None:
                # If no weights provided, this simply calls `np.percentile`.
                return np.percentile(x, list(100.0 * q))
            else:
                # If weights are provided, compute the weighted quantiles.
                weights = np.atleast_1d(weights)
                if len(x) != len(weights):
                    raise ValueError("Dimension mismatch: len(weights) != len(x).")
                idx = np.argsort(x)  # sort samples
                sw = weights[idx]  # sort weights
                cdf = np.cumsum(sw)[:-1]  # compute CDF
                cdf /= cdf[-1]  # normalize CDF
                cdf = np.append(0, cdf)  # ensure proper span
                quantiles = np.interp(q, cdf, x[idx]).tolist()
                return quantiles

        def _store_nest_solutions():
            NEST_out = {'solutions': {}}
            data = np.loadtxt(prefix + '.txt')
            NEST_stats = multinest_results.get_stats()
            NEST_out['NEST_stats'] = NEST_stats
            NEST_out['global_logE'] = (NEST_out['NEST_stats']['global evidence'], NEST_out['NEST_stats']['global evidence error'])

            modes = []
            modes_weights = []
            modes_loglike = []
            chains = []
            chains_weights = []
            chains_loglike = []

            if self.param['multimodal'] and mds_orig > 1:
                # separate modes. get individual samples for each mode
                # get parameter values and sample probability (=weight) for each mode
                with open(prefix + 'post_separate.dat') as f:
                    lines = f.readlines()
                    for idx, line in enumerate(lines):
                        if idx > 2:  # skip the first two lines
                            if lines[idx - 1] == '\n' and lines[idx - 2] == '\n':
                                modes.append(chains)
                                modes_weights.append(chains_weights)
                                modes_loglike.append(chains_loglike)
                                chains = []
                                chains_weights = []
                                chains_loglike = []
                        chain = [float(x) for x in line.split()[2:]]
                        if len(chain) > 0:
                            chains.append(chain)
                            chains_weights.append(float(line.split()[0]))
                            chains_loglike.append(float(line.split()[1]))
                    modes.append(chains)
                    modes_weights.append(chains_weights)
                    modes_loglike.append(chains_loglike)
                modes_array = []
                for mode in modes:
                    mode_array = np.zeros((len(mode), len(mode[0])))
                    for idx, line in enumerate(mode):
                        mode_array[idx, :] = line
                    modes_array.append(mode_array)
            else:
                # not running in multimode. Get chains directly from file prefix.txt
                modes_array = [data[:, 2:]]
                chains_weights = [data[:, 0]]
                modes_weights.append(chains_weights[0])
                chains_loglike = [data[:, 1]]
                modes_loglike.append(chains_loglike[0])
                modes = [0]

            for nmode in range(len(modes)):
                mydict = {'type': 'nest',
                          'local_logE': (NEST_out['NEST_stats']['modes'][nmode]['local log-evidence'], NEST_out['NEST_stats']['modes'][nmode]['local log-evidence error']),
                          'weights': np.asarray(modes_weights[nmode]),
                          'loglike': np.asarray(modes_loglike[nmode]),
                          'tracedata': modes_array[nmode],
                          'fit_params': {}}

                for idx, param_name in enumerate(parameters):
                    trace = modes_array[nmode][:, idx]
                    q_16, q_50, q_84 = _quantile(trace, [0.16, 0.5, 0.84], weights=np.asarray(modes_weights[nmode]))
                    mydict['fit_params'][param_name] = {
                        'value': q_50,
                        'sigma_m': q_50 - q_16,
                        'sigma_p': q_84 - q_50,
                        'nest_map': NEST_stats['modes'][nmode]['maximum a posterior'][idx],
                        'mean': NEST_stats['modes'][nmode]['mean'][idx],
                        'nest_sigma': NEST_stats['modes'][nmode]['sigma'][idx],
                        'trace': trace,
                    }

                NEST_out['solutions']['solution{}'.format(nmode)] = mydict

            if len(NEST_out['solutions']) > 1:
                for i in range(len(NEST_out['solutions'])):
                    fl = np.ones((len(NEST_out['solutions']['solution' + str(i)]['weights']), len(NEST_out['solutions']['solution' + str(i)]['tracedata'][0, :]) + 2))
                    fl[:, 0] = NEST_out['solutions']['solution' + str(i)]['weights']
                    fl[:, 1] = NEST_out['solutions']['solution' + str(i)]['loglike']
                    fl[:, 2:] = NEST_out['solutions']['solution' + str(i)]['tracedata']
                    np.savetxt(prefix + 'solution' + str(i) + '.txt', fl)

            return NEST_out

        def _plotting_bounds(results, modes=None):
            if modes is None:
                samples = results['samples']
                try:
                    weights = np.exp(results['logwt'] - results['logz'][-1])
                except:
                    weights = results['weights']

                # Deal with 1D results. A number of extra catches are also here
                # in case users are trying to plot other results besides the `Results`
                # instance generated by `dynesty`.
                samples = np.atleast_1d(samples)
                if len(samples.shape) == 1:
                    samples = np.atleast_2d(samples)
                else:
                    assert len(samples.shape) == 2, "Samples must be 1- or 2-D."
                    samples = samples.T
                assert samples.shape[0] <= samples.shape[1], "There are more " \
                                                             "dimensions than samples!"
                ndim, nsamps = samples.shape

                # Check weights.
                if weights.ndim != 1:
                    raise ValueError("Weights must be 1-D.")
                if nsamps != weights.shape[0]:
                    raise ValueError("The number of weights and samples disagree!")

                boundaries = [0.999999426697 for _ in range(len(results['samples'][0, :]))]
                boundaries = list(boundaries)
                for i, _ in enumerate(boundaries):
                    q = [0.5 - 0.5 * boundaries[i], 0.5 + 0.5 * boundaries[i]]
                    boundaries[i] = _quantile(samples[i], q, weights=weights)

                return boundaries
            else:
                boundaries = {}
                for sol in range(0, modes):
                    # Extract weighted samples.
                    samples = results[str(sol)]['samples']
                    try:
                        weights = np.exp(results[str(sol)]['logwt'] - results[str(sol)]['logz'][-1])
                    except:
                        weights = results[str(sol)]['weights']

                    # Deal with 1D results. A number of extra catches are also here
                    # in case users are trying to plot other results besides the `Results`
                    # instance generated by `dynesty`.
                    samples = np.atleast_1d(samples)
                    if len(samples.shape) == 1:
                        samples = np.atleast_2d(samples)
                    else:
                        assert len(samples.shape) == 2, "Samples must be 1- or 2-D."
                        samples = samples.T
                    assert samples.shape[0] <= samples.shape[1], "There are more " \
                                                                 "dimensions than samples!"
                    ndim, nsamps = samples.shape

                    # Check weights.
                    if weights.ndim != 1:
                        raise ValueError("Weights must be 1-D.")
                    if nsamps != weights.shape[0]:
                        raise ValueError("The number of weights and samples disagree!")

                    boundaries[str(sol)] = [0.999999426697 for _ in range(len(results[str(sol)]['samples'][0, :]))]
                    boundaries[str(sol)] = list(boundaries[str(sol)])
                    for i, _ in enumerate(boundaries[str(sol)]):
                        q = [0.5 - 0.5 * boundaries[str(sol)][i], 0.5 + 0.5 * boundaries[str(sol)][i]]
                        boundaries[str(sol)][i] = _quantile(samples[i], q, weights=weights)

                bound = []
                for i in range(ndim):
                    min_b, max_b = [], []
                    for j in boundaries.keys():
                        min_b.append(boundaries[j][i][0])
                        max_b.append(boundaries[j][i][1])
                    bound.append([min(min_b), max(max_b)])

                return list(bound)

        def _resample_equal(samples, weights, rstate=None):
            """
            Resample a new set of points from the weighted set of inputs
            such that they all have equal weight.
            Each input sample appears in the output array either
            `floor(weights[i] * nsamples)` or `ceil(weights[i] * nsamples)` times,
            with `floor` or `ceil` randomly selected (weighted by proximity).
            Parameters
            ----------
            samples : `~numpy.ndarray` with shape (nsamples,)
                Set of unequally weighted samples.
            weights : `~numpy.ndarray` with shape (nsamples,)
                Corresponding weight of each sample.
            rstate : `~numpy.random.RandomState`, optional
                `~numpy.random.RandomState` instance.
            Returns
            -------
            equal_weight_samples : `~numpy.ndarray` with shape (nsamples,)
                New set of samples with equal weights.
            Notes
            -----
            Implements the systematic resampling method described in `Hol, Schon, and
            Gustafsson (2006) <doi:10.1109/NSSPW.2006.4378824>`_.
            """

            if rstate is None:
                rstate = np.random

            if abs(np.sum(weights) - 1.) > SQRTEPS:  # same tol as in np.random.choice.
                raise ValueError("Weights do not sum to 1.")

            # Make N subdivisions and choose positions with a consistent random offset.
            nsamples = len(weights)
            positions = (rstate.random() + np.arange(nsamples)) / nsamples

            # Resample the data.
            idx = np.zeros(nsamples, dtype=int)
            cumulative_sum = np.cumsum(weights)
            i, j = 0, 0
            while i < nsamples:
                if positions[i] < cumulative_sum[j]:
                    idx[i] = j
                    i += 1
                else:
                    j += 1

            return samples[idx]

        def _hist2d(x, y, smooth=0.02, span=None, weights=None, levels=None,
                    ax=None, color='gray', plot_datapoints=False, plot_density=True,
                    plot_contours=True, no_fill_contours=False, fill_contours=True,
                    contour_kwargs=None, contourf_kwargs=None, data_kwargs=None,
                    **kwargs):
            """
            Internal function called by :meth:`cornerplot` used to generate a
            a 2-D histogram/contour of samples.

            Parameters
            ----------
            x : interable with shape (nsamps,)
               Sample positions in the first dimension.

            y : iterable with shape (nsamps,)
               Sample positions in the second dimension.

            span : iterable with shape (ndim,), optional
                A list where each element is either a length-2 tuple containing
                lower and upper bounds or a float from `(0., 1.]` giving the
                fraction of (weighted) samples to include. If a fraction is provided,
                the bounds are chosen to be equal-tailed. An example would be::

                    span = [(0., 10.), 0.95, (5., 6.)]

                Default is `0.999999426697` (5-sigma credible interval).

            weights : iterable with shape (nsamps,)
                Weights associated with the samples. Default is `None` (no weights).

            levels : iterable, optional
                The contour levels to draw. Defaults correspond to
                `[0.5, 1.0, 2.0, 3.0]`-sigma for a 2-D Gaussian, i.e.
                mass fractions `1 - exp(-sigma^2/2)`.

            ax : `~matplotlib.axes.Axes`, optional
                An `~matplotlib.axes.axes` instance on which to add the 2-D histogram.
                If not provided, a figure will be generated.

            color : str, optional
                The `~matplotlib`-style color used to draw lines and color cells
                and contours. Default is `'gray'`.

            plot_datapoints : bool, optional
                Whether to plot the individual data points. Default is `False`.

            plot_density : bool, optional
                Whether to draw the density colormap. Default is `True`.

            plot_contours : bool, optional
                Whether to draw the contours. Default is `True`.

            no_fill_contours : bool, optional
                Whether to add absolutely no filling to the contours. This differs
                from `fill_contours=False`, which still adds a white fill at the
                densest points. Default is `False`.

            fill_contours : bool, optional
                Whether to fill the contours. Default is `True`.

            contour_kwargs : dict
                Any additional keyword arguments to pass to the `contour` method.

            contourf_kwargs : dict
                Any additional keyword arguments to pass to the `contourf` method.

            data_kwargs : dict
                Any additional keyword arguments to pass to the `plot` method when
                adding the individual data points.

            """

            if ax is None:
                ax = plt.gca()

            # Determine plotting bounds.
            data = [x, y]
            if span is None:
                span = [0.999999426697 for i in range(2)]
            span = list(span)
            if len(span) != 2:
                raise ValueError("Dimension mismatch between samples and span.")
            for i, _ in enumerate(span):
                try:
                    xmin, xmax = span[i]
                except:
                    q = [0.5 - 0.5 * span[i], 0.5 + 0.5 * span[i]]
                    span[i] = _quantile(data[i], q, weights=weights)

            # Default contour levels at 0.5, 1.0, 2.0, 3.0 sigma.
            # For a 2-D Gaussian, enclosed mass at radius n-sigma is 1 - exp(-n^2/2).
            if levels is None:
                sigmas = np.array([0.5, 1.0, 2.0, 3.0])
                levels = 1.0 - np.exp(-0.5 * sigmas ** 2)

            # Color map for the density plot, over-plotted to indicate the
            # density of the points near the center.
            density_cmap = LinearSegmentedColormap.from_list(
                "density_cmap", [color, (1, 1, 1, 0)])

            # Color map used to hide the points at the high density areas.
            white_cmap = LinearSegmentedColormap.from_list(
                "white_cmap", [(1, 1, 1), (1, 1, 1)], N=2)

            # This "color map" is the list of colors for the contour levels if the
            # contours are filled.
            rgba_color = colorConverter.to_rgba(color)
            contour_cmap = [list(rgba_color) for l in levels] + [rgba_color]
            for i, l in enumerate(levels):
                contour_cmap[i][-1] *= float(i) / (len(levels) + 1)

            # Initialize smoothing.
            if (isinstance(smooth, int_type) or isinstance(smooth, float_type)):
                smooth = [smooth, smooth]
            bins = []
            svalues = []
            for s in smooth:
                if isinstance(s, int_type):
                    # If `s` is an integer, the weighted histogram has
                    # `s` bins within the provided bounds.
                    bins.append(s)
                    svalues.append(0.)
                else:
                    # If `s` is a float, oversample the data relative to the
                    # smoothing filter by a factor of 2, then use a Gaussian
                    # filter to smooth the results.
                    bins.append(int(round(2. / s)))
                    svalues.append(2.)

            # We'll make the 2D histogram to directly estimate the density.
            try:
                H, X, Y = np.histogram2d(x.flatten(), y.flatten(), bins=bins,
                                         range=list(map(np.sort, span)),
                                         weights=weights)
            except ValueError:
                raise ValueError("It looks like at least one of your sample columns "
                                 "have no dynamic range.")

            # Smooth the results.
            if not np.all(svalues == 0.):
                H = norm_kde(H, svalues)

            # Compute the density levels.
            Hflat = H.flatten()
            inds = np.argsort(Hflat)[::-1]
            Hflat = Hflat[inds]
            sm = np.cumsum(Hflat)
            sm /= sm[-1]
            V = np.empty(len(levels))
            for i, v0 in enumerate(levels):
                try:
                    V[i] = Hflat[sm <= v0][-1]
                except:
                    V[i] = Hflat[0]
            V.sort()
            m = (np.diff(V) == 0)
            if np.any(m) and plot_contours:
                logging.warning("Too few points to create valid contours.")
            while np.any(m):
                V[np.where(m)[0][0]] *= 1.0 - 1e-4
                m = (np.diff(V) == 0)
            V.sort()

            # Compute the bin centers.
            X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

            # Extend the array for the sake of the contours at the plot edges.
            H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
            H2[2:-2, 2:-2] = H
            H2[2:-2, 1] = H[:, 0]
            H2[2:-2, -2] = H[:, -1]
            H2[1, 2:-2] = H[0]
            H2[-2, 2:-2] = H[-1]
            H2[1, 1] = H[0, 0]
            H2[1, -2] = H[0, -1]
            H2[-2, 1] = H[-1, 0]
            H2[-2, -2] = H[-1, -1]
            X2 = np.concatenate([X1[0] + np.array([-2, -1]) * np.diff(X1[:2]), X1,
                                 X1[-1] + np.array([1, 2]) * np.diff(X1[-2:])])
            Y2 = np.concatenate([Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]), Y1,
                                 Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:])])

            # Plot the data points.
            if plot_datapoints:
                if data_kwargs is None:
                    data_kwargs = dict()
                data_kwargs["color"] = data_kwargs.get("color", color)
                data_kwargs["ms"] = data_kwargs.get("ms", 2.0)
                data_kwargs["mec"] = data_kwargs.get("mec", "none")
                data_kwargs["alpha"] = data_kwargs.get("alpha", 0.1)
                ax.plot(x, y, "o", zorder=-1, rasterized=True, **data_kwargs)

            # Plot the base fill to hide the densest data points.
            if (plot_contours or plot_density) and not no_fill_contours:
                ax.contourf(X2, Y2, H2.T, [V.min(), H.max()],
                            cmap=white_cmap, antialiased=False)

            if plot_contours and fill_contours:
                if contourf_kwargs is None:
                    contourf_kwargs = dict()
                contourf_kwargs["colors"] = contourf_kwargs.get("colors", contour_cmap)
                contourf_kwargs["antialiased"] = contourf_kwargs.get("antialiased",
                                                                     False)
                ax.contourf(X2, Y2, H2.T, np.concatenate([[0], V, [H.max() * (1 + 1e-4)]]),
                            **contourf_kwargs)

            # Plot the density map. This can't be plotted at the same time as the
            # contour fills.
            elif plot_density:
                ax.pcolor(X, Y, H.max() - H.T, cmap=density_cmap)

            # Plot the contour edge colors.
            if plot_contours:
                if contour_kwargs is None:
                    contour_kwargs = dict()
                contour_kwargs["colors"] = contour_kwargs.get("colors", color)
                ax.contour(X2, Y2, H2.T, V, **contour_kwargs)

            ax.set_xlim(span[0])
            ax.set_ylim(span[1])

        def traceplot(results, span=None, quantiles=[0.16, 0.5, 0.84], smooth=0.02,
                      post_color='blue', post_kwargs=None, kde=True, nkde=1000,
                      trace_cmap='plasma', trace_color=None, trace_kwargs=None,
                      connect=False, connect_highlight=10, connect_color='red',
                      connect_kwargs=None, max_n_ticks=5, use_math_text=False,
                      labels=None, label_kwargs=None,
                      show_titles=False, title_fmt=".2f", title_kwargs=None,
                      truths=None, truth_color='red', truth_kwargs=None,
                      verbose=False, fig=None):
            """
            Plot traces and marginalized posteriors for each parameter.

            Parameters
            ----------
            results : :class:`~dynesty.results.Results` instance
                A :class:`~dynesty.results.Results` instance from a nested
                sampling run. **Compatible with results derived from**
                `nestle <http://kylebarbary.com/nestle/>`_.

            span : iterable with shape (ndim,), optional
                A list where each element is either a length-2 tuple containing
                lower and upper bounds or a float from `(0., 1.]` giving the
                fraction of (weighted) samples to include. If a fraction is provided,
                the bounds are chosen to be equal-tailed. An example would be::

                    span = [(0., 10.), 0.95, (5., 6.)]

                Default is `0.999999426697` (5-sigma credible interval) for each
                parameter.

            quantiles : iterable, optional
                A list of fractional quantiles to overplot on the 1-D marginalized
                posteriors as vertical dashed lines. Default is `[0.16, 0.5, 0.84]`
                (the 68%/1-sigma credible interval). Use `[0.0225, 0.5, 0.9775]`
                for 95.5%/2-sigma credible interval, and `[0.00135, 0.5, 0.99865] for
                99.73%/3-sigma.

            smooth : float or iterable with shape (ndim,), optional
                The standard deviation (either a single value or a different value for
                each subplot) for the Gaussian kernel used to smooth the 1-D
                marginalized posteriors, expressed as a fraction of the span.
                Default is `0.02` (2% smoothing). If an integer is provided instead,
                this will instead default to a simple (weighted) histogram with
                `bins=smooth`.

            post_color : str or iterable with shape (ndim,), optional
                A `~matplotlib`-style color (either a single color or a different
                value for each subplot) used when plotting the histograms.
                Default is `'blue'`.

            post_kwargs : dict, optional
                Extra keyword arguments that will be used for plotting the
                marginalized 1-D posteriors.

            kde : bool, optional
                Whether to use kernel density estimation to estimate and plot
                the PDF of the importance weights as a function of log-volume
                (as opposed to the importance weights themselves). Default is
                `True`.

            nkde : int, optional
                The number of grid points used when plotting the kernel density
                estimate. Default is `1000`.

            trace_cmap : str or iterable with shape (ndim,), optional
                A `~matplotlib`-style colormap (either a single colormap or a
                different colormap for each subplot) used when plotting the traces,
                where each point is colored according to its weight. Default is
                `'plasma'`.

            trace_color : str or iterable with shape (ndim,), optional
                A `~matplotlib`-style color (either a single color or a
                different color for each subplot) used when plotting the traces.
                This overrides the `trace_cmap` option by giving all points
                the same color. Default is `None` (not used).

            trace_kwargs : dict, optional
                Extra keyword arguments that will be used for plotting the traces.

            connect : bool, optional
                Whether to draw lines connecting the paths of unique particles.
                Default is `False`.

            connect_highlight : int or iterable, optional
                If `connect=True`, highlights the paths of a specific set of
                particles. If an integer is passed, :data:`connect_highlight`
                random particle paths will be highlighted. If an iterable is passed,
                then the particle paths corresponding to the provided indices
                will be highlighted.

            connect_color : str, optional
                The color of the highlighted particle paths. Default is `'red'`.

            connect_kwargs : dict, optional
                Extra keyword arguments used for plotting particle paths.

            max_n_ticks : int, optional
                Maximum number of ticks allowed. Default is `5`.

            use_math_text : bool, optional
                Whether the axis tick labels for very large/small exponents should be
                displayed as powers of 10 rather than using `e`. Default is `False`.

            labels : iterable with shape (ndim,), optional
                A list of names for each parameter. If not provided, the default name
                used when plotting will follow :math:`x_i` style.

            label_kwargs : dict, optional
                Extra keyword arguments that will be sent to the
                `~matplotlib.axes.Axes.set_xlabel` and
                `~matplotlib.axes.Axes.set_ylabel` methods.

            show_titles : bool, optional
                Whether to display a title above each 1-D marginalized posterior
                showing the 0.5 quantile along with the upper/lower bounds associated
                with the 0.025 and 0.975 (95%/2-sigma credible interval) quantiles.
                Default is `True`.

            title_fmt : str, optional
                The format string for the quantiles provided in the title. Default is
                `'.2f'`.

            title_kwargs : dict, optional
                Extra keyword arguments that will be sent to the
                `~matplotlib.axes.Axes.set_title` command.

            truths : iterable with shape (ndim,), optional
                A list of reference values that will be overplotted on the traces and
                marginalized 1-D posteriors as solid horizontal/vertical lines.
                Individual values can be exempt using `None`. Default is `None`.

            truth_color : str or iterable with shape (ndim,), optional
                A `~matplotlib`-style color (either a single color or a different
                value for each subplot) used when plotting `truths`.
                Default is `'red'`.

            truth_kwargs : dict, optional
                Extra keyword arguments that will be used for plotting the vertical
                and horizontal lines with `truths`.

            verbose : bool, optional
                Whether to print the values of the computed quantiles associated with
                each parameter. Default is `False`.

            fig : (`~matplotlib.figure.Figure`, `~matplotlib.axes.Axes`), optional
                If provided, overplot the traces and marginalized 1-D posteriors
                onto the provided figure. Otherwise, by default an
                internal figure is generated.

            Returns
            -------
            traceplot : (`~matplotlib.figure.Figure`, `~matplotlib.axes.Axes`)
                Output trace plot.

            """

            # Initialize values.
            if title_kwargs is None:
                title_kwargs = dict()
            if label_kwargs is None:
                label_kwargs = dict()
            if trace_kwargs is None:
                trace_kwargs = dict()
            if connect_kwargs is None:
                connect_kwargs = dict()
            if post_kwargs is None:
                post_kwargs = dict()
            if truth_kwargs is None:
                truth_kwargs = dict()

            # Set defaults.
            connect_kwargs['alpha'] = connect_kwargs.get('alpha', 0.7)
            post_kwargs['alpha'] = post_kwargs.get('alpha', 0.6)
            trace_kwargs['s'] = trace_kwargs.get('s', 3)
            truth_kwargs['linestyle'] = truth_kwargs.get('linestyle', 'solid')
            truth_kwargs['linewidth'] = truth_kwargs.get('linewidth', 2)

            # Extract weighted samples.
            samples = results['samples']
            logvol = results['logvol']
            try:
                weights = np.exp(results['logwt'] - results['logz'][-1])
            except:
                weights = results['weights']
            if kde:
                # Derive kernel density estimate.
                wt_kde = gaussian_kde(_resample_equal(-logvol, weights))  # KDE
                logvol_grid = np.linspace(logvol[0], logvol[-1], nkde)  # resample
                wt_grid = wt_kde.pdf(-logvol_grid)  # evaluate KDE PDF
                wts = np.interp(-logvol, -logvol_grid, wt_grid)  # interpolate
            else:
                wts = weights

            # Deal with 1D results. A number of extra catches are also here
            # in case users are trying to plot other results besides the `Results`
            # instance generated by `dynesty`.
            samples = np.atleast_1d(samples)
            if len(samples.shape) == 1:
                samples = np.atleast_2d(samples)
            else:
                assert len(samples.shape) == 2, "Samples must be 1- or 2-D."
                samples = samples.T
            assert samples.shape[0] <= samples.shape[1], "There are more " \
                                                         "dimensions than samples!"
            ndim, nsamps = samples.shape

            # Check weights.
            if weights.ndim != 1:
                raise ValueError("Weights must be 1-D.")
            if nsamps != weights.shape[0]:
                raise ValueError("The number of weights and samples disagree!")

            # Check ln(volume).
            if logvol.ndim != 1:
                raise ValueError("Ln(volume)'s must be 1-D.")
            if nsamps != logvol.shape[0]:
                raise ValueError("The number of ln(volume)'s and samples disagree!")

            # Check sample IDs.
            if connect:
                try:
                    samples_id = results['samples_id']
                    uid = np.unique(samples_id)
                except:
                    raise ValueError("Sample IDs are not defined!")
                try:
                    ids = connect_highlight[0]
                    ids = connect_highlight
                except:
                    ids = np.random.choice(uid, size=connect_highlight, replace=False)

            # Determine plotting bounds for marginalized 1-D posteriors.
            if span is None:
                span = [0.999999426697 for i in range(ndim)]
            span = list(span)
            if len(span) != ndim:
                raise ValueError("Dimension mismatch between samples and span.")
            for i, _ in enumerate(span):
                try:
                    xmin, xmax = span[i]
                except:
                    q = [0.5 - 0.5 * span[i], 0.5 + 0.5 * span[i]]
                    span[i] = _quantile(samples[i], q, weights=weights)

            # Setting up labels.
            if labels is None:
                labels = [r"$x_{" + str(i + 1) + "}$" for i in range(ndim)]

            # Setting up smoothing.
            if (isinstance(smooth, int_type) or isinstance(smooth, float_type)):
                smooth = [smooth for i in range(ndim)]

            # Setting up default plot layout.
            if fig is None:
                fig, axes = plt.subplots(ndim, 2, figsize=(12, 3 * ndim))
            else:
                fig, axes = fig
                try:
                    axes.reshape(ndim, 2)
                except:
                    raise ValueError("Provided axes do not match the required shape "
                                     "for plotting samples.")

            # Plotting.
            for i, x in enumerate(samples):

                # Plot trace.

                # Establish axes.
                if np.shape(samples)[0] == 1:
                    ax = axes[1]
                else:
                    ax = axes[i, 0]
                # Set color(s)/colormap(s).
                if trace_color is not None:
                    if isinstance(trace_color, str_type):
                        color = trace_color
                    else:
                        color = trace_color[i]
                else:
                    color = wts
                if isinstance(trace_cmap, str_type):
                    cmap = trace_cmap
                else:
                    cmap = trace_cmap[i]
                # Setup axes.
                ax.set_xlim([0., -min(logvol)])
                ax.set_ylim([min(x), max(x)])
                if max_n_ticks == 0:
                    ax.xaxis.set_major_locator(NullLocator())
                    ax.yaxis.set_major_locator(NullLocator())
                else:
                    ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks))
                    ax.yaxis.set_major_locator(MaxNLocator(max_n_ticks))
                # Label axes.
                sf = ScalarFormatter(useMathText=use_math_text)
                ax.yaxis.set_major_formatter(sf)
                ax.set_xlabel(r"$-\ln X$", **label_kwargs)
                ax.set_ylabel(labels[i], **label_kwargs)
                # Generate scatter plot.
                ax.scatter(-logvol, x, c=color, cmap=cmap, **trace_kwargs)
                if connect:
                    # Add lines highlighting specific particle paths.
                    for j in ids:
                        sel = (samples_id == j)
                        ax.plot(-logvol[sel], x[sel], color=connect_color,
                                **connect_kwargs)
                # Add truth value(s).
                if truths is not None and truths[i] is not None:
                    try:
                        [ax.axhline(t, color=truth_color, **truth_kwargs)
                         for t in truths[i]]
                    except:
                        ax.axhline(truths[i], color=truth_color, **truth_kwargs)

                # Plot marginalized 1-D posterior.

                # Establish axes.
                if np.shape(samples)[0] == 1:
                    ax = axes[0]
                else:
                    ax = axes[i, 1]
                # Set color(s).
                if isinstance(post_color, str_type):
                    color = post_color
                else:
                    color = post_color[i]
                # Setup axes
                ax.set_xlim(span[i])
                if max_n_ticks == 0:
                    ax.xaxis.set_major_locator(NullLocator())
                    ax.yaxis.set_major_locator(NullLocator())
                else:
                    ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks))
                    ax.yaxis.set_major_locator(NullLocator())
                # Label axes.
                sf = ScalarFormatter(useMathText=use_math_text)
                ax.xaxis.set_major_formatter(sf)
                ax.set_xlabel(labels[i], **label_kwargs)
                # Generate distribution.
                s = smooth[i]
                if isinstance(s, int_type):
                    # If `s` is an integer, plot a weighted histogram with
                    # `s` bins within the provided bounds.
                    n, b, _ = ax.hist(x, bins=s, weights=weights, color=color,
                                      range=np.sort(span[i]), **post_kwargs)
                    x0 = np.array(list(zip(b[:-1], b[1:]))).flatten()
                    y0 = np.array(list(zip(n, n))).flatten()
                else:
                    # If `s` is a float, oversample the data relative to the
                    # smoothing filter by a factor of 10, then use a Gaussian
                    # filter to smooth the results.
                    bins = int(round(10. / s))
                    n, b = np.histogram(x, bins=bins, weights=weights,
                                        range=np.sort(span[i]))
                    n = norm_kde(n, 10.)
                    x0 = 0.5 * (b[1:] + b[:-1])
                    y0 = n
                    ax.fill_between(x0, y0, color=color, **post_kwargs)
                ax.set_ylim([0., max(y0) * 1.05])
                # Plot quantiles.
                if quantiles is not None and len(quantiles) > 0:
                    qs = _quantile(x, quantiles, weights=weights)
                    for q in qs:
                        ax.axvline(q, lw=1, ls="dashed", color=color)
                    if verbose:
                        print("Quantiles:")
                        print(labels[i], [blob for blob in zip(quantiles, qs)])
                # Add truth value(s).
                if truths is not None and truths[i] is not None:
                    try:
                        [ax.axvline(t, color=truth_color, **truth_kwargs)
                         for t in truths[i]]
                    except:
                        ax.axvline(truths[i], color=truth_color, **truth_kwargs)
                # Set titles.
                if show_titles:
                    title = None
                    if title_fmt is not None:
                        ql, qm, qh = _quantile(x, [0.16, 0.5, 0.84], weights=weights)
                        q_minus, q_plus = qm - ql, qh - qm
                        fmt = "{{0:{0}}}".format(title_fmt).format
                        title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
                        title = title.format(fmt(qm), fmt(q_minus), fmt(q_plus))
                        title = "{0} = {1}".format(labels[i], title)
                        ax.set_title(title, **title_kwargs)

            return fig, axes


        # 2D marginalized default contour levels handled in `_hist2d` (0.5, 1.0, 2.0, 3.0 sigma).
        # Corner plot with lighter (less aggressive) 1D smoothing.
        def cornerplot(results, span=None, quantiles=[0.16, 0.5, 0.84],
                       color='#404784', smooth=0.02, hist_kwargs=None,
                       hist2d_kwargs=None, labels=None, label_kwargs=None,
                       show_titles=False, title_fmt=".2f", title_kwargs=None,
                       truths=None, truth_color='red', truth_kwargs=None,
                       max_n_ticks=5, top_ticks=False, use_math_text=False,
                       verbose=False, fig=None):
            # type: (object, object, object, object, object, object, object, object, object, object, object, object, object, object, object, object, object, object, object, object) -> object
            """
            Generate a corner plot of the 1-D and 2-D marginalized posteriors.

            Parameters
            ----------
            results : :class:`~dynesty.results.Results` instance
                A :class:`~dynesty.results.Results` instance from a nested
                sampling run. **Compatible with results derived from**
                `nestle <http://kylebarbary.com/nestle/>`_.

            span : iterable with shape (ndim,), optional
                A list where each element is either a length-2 tuple containing
                lower and upper bounds or a float from `(0., 1.]` giving the
                fraction of (weighted) samples to include. If a fraction is provided,
                the bounds are chosen to be equal-tailed. An example would be::

                    span = [(0., 10.), 0.95, (5., 6.)]

                Default is `0.999999426697` (5-sigma credible interval).

            quantiles : iterable, optional
                A list of fractional quantiles to overplot on the 1-D marginalized
                posteriors as vertical dashed lines. Default is `[0.16, 0.5, 0.84]`
                (spanning the 68%/1-sigma credible interval). Use `[0.0225, 0.5, 0.9775]`
                for 95%/2-sigma credible interval, and `[0.00135, 0.5, 0.99865] for
                99.73%/3-sigma.

            color : str or iterable with shape (ndim,), optional
                A `~matplotlib`-style color (either a single color or a different
                value for each subplot) used when plotting the histograms.
                Default is `'black'`.

            smooth : float or iterable with shape (ndim,), optional
                The standard deviation (either a single value or a different value for
                each subplot) for the Gaussian kernel used to smooth the 1-D and 2-D
                marginalized posteriors, expressed as a fraction of the span.
                Default is `0.02` (2% smoothing). If an integer is provided instead,
                this will instead default to a simple (weighted) histogram with
                `bins=smooth`.

            hist_kwargs : dict, optional
                Extra keyword arguments to send to the 1-D (smoothed) histograms.

            hist2d_kwargs : dict, optional
                Extra keyword arguments to send to the 2-D (smoothed) histograms.

            labels : iterable with shape (ndim,), optional
                A list of names for each parameter. If not provided, the default name
                used when plotting will follow :math:`x_i` style.

            label_kwargs : dict, optional
                Extra keyword arguments that will be sent to the
                `~matplotlib.axes.Axes.set_xlabel` and
                `~matplotlib.axes.Axes.set_ylabel` methods.

            show_titles : bool, optional
                Whether to display a title above each 1-D marginalized posterior
                showing the 0.5 quantile along with the upper/lower bounds associated
                with the 0.025 and 0.975 (95%/2-sigma credible interval) quantiles.
                Default is `True`.

            title_fmt : str, optional
                The format string for the quantiles provided in the title. Default is
                `'.2f'`.

            title_kwargs : dict, optional
                Extra keyword arguments that will be sent to the
                `~matplotlib.axes.Axes.set_title` command.

            truths : iterable with shape (ndim,), optional
                A list of reference values that will be overplotted on the traces and
                marginalized 1-D posteriors as solid horizontal/vertical lines.
                Individual values can be exempt using `None`. Default is `None`.

            truth_color : str or iterable with shape (ndim,), optional
                A `~matplotlib`-style color (either a single color or a different
                value for each subplot) used when plotting `truths`.
                Default is `'red'`.

            truth_kwargs : dict, optional
                Extra keyword arguments that will be used for plotting the vertical
                and horizontal lines with `truths`.

            max_n_ticks : int, optional
                Maximum number of ticks allowed. Default is `5`.

            top_ticks : bool, optional
                Whether to label the top (rather than bottom) ticks. Default is
                `False`.

            use_math_text : bool, optional
                Whether the axis tick labels for very large/small exponents should be
                displayed as powers of 10 rather than using `e`. Default is `False`.

            verbose : bool, optional
                Whether to print the values of the computed quantiles associated with
                each parameter. Default is `False`.

            fig : (`~matplotlib.figure.Figure`, `~matplotlib.axes.Axes`), optional
                If provided, overplot the traces and marginalized 1-D posteriors
                onto the provided figure. Otherwise, by default an
                internal figure is generated.

            Returns
            -------
            cornerplot : (`~matplotlib.figure.Figure`, `~matplotlib.axes.Axes`)
                Output corner plot.

            """

            # Initialize values.
            if quantiles is None:
                quantiles = []
            if truth_kwargs is None:
                truth_kwargs = dict()
            if label_kwargs is None:
                label_kwargs = dict()
            if title_kwargs is None:
                title_kwargs = dict()
            if hist_kwargs is None:
                hist_kwargs = dict()
            if hist2d_kwargs is None:
                hist2d_kwargs = dict()

            # Set defaults.
            hist_kwargs['alpha'] = hist_kwargs.get('alpha', 0.6)
            hist2d_kwargs['alpha'] = hist2d_kwargs.get('alpha', 0.6)
            truth_kwargs['linestyle'] = truth_kwargs.get('linestyle', 'solid')
            truth_kwargs['linewidth'] = truth_kwargs.get('linewidth', 2)
            truth_kwargs['alpha'] = truth_kwargs.get('alpha', 0.7)

            # Extract weighted samples.
            samples = results['samples']
            try:
                weights = np.exp(results['logwt'] - results['logz'][-1])
            except:
                weights = results['weights']

            # Deal with 1D results. A number of extra catches are also here
            # in case users are trying to plot other results besides the `Results`
            # instance generated by `dynesty`.
            samples = np.atleast_1d(samples)
            if len(samples.shape) == 1:
                samples = np.atleast_2d(samples)
            else:
                assert len(samples.shape) == 2, "Samples must be 1- or 2-D."
                samples = samples.T
            assert samples.shape[0] <= samples.shape[1], "There are more " \
                                                         "dimensions than samples!"
            ndim, nsamps = samples.shape

            # Check weights.
            if weights.ndim != 1:
                raise ValueError("Weights must be 1-D.")
            if nsamps != weights.shape[0]:
                raise ValueError("The number of weights and samples disagree!")

            # Set labels
            if labels is None:
                labels = [r"$x_{" + str(i + 1) + "}$" for i in range(ndim)]

            # Setting up smoothing.
            if (isinstance(smooth, int_type) or isinstance(smooth, float_type)):
                smooth = [smooth for i in range(ndim)]

            # Setup axis layout (from `corner.py`).
            factor = 2.0  # size of side of one panel
            lbdim = 0.5 * factor  # size of left/bottom margin
            trdim = 0.2 * factor  # size of top/right margin
            whspace = 0.05  # size of width/height margin
            plotdim = factor * ndim + factor * (ndim - 1.) * whspace  # plot size
            dim = lbdim + plotdim + trdim  # total size

            # Initialize figure.
            if fig is None:
                fig, axes = plt.subplots(ndim, ndim, figsize=(dim, dim))
            else:
                try:
                    fig, axes = fig
                    axes = np.array(axes).reshape((ndim, ndim))
                except:
                    raise ValueError("Mismatch between axes and dimension.")

            # Format figure.
            lb = lbdim / dim
            tr = (lbdim + plotdim) / dim
            fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr,
                                wspace=whspace, hspace=whspace)

            # Plotting.
            for i, x in enumerate(samples):
                if np.shape(samples)[0] == 1:
                    ax = axes
                else:
                    ax = axes[i, i]

                # Plot the 1-D marginalized posteriors.

                # Setup axes
                ax.set_xlim(span[i])
                if max_n_ticks == 0:
                    ax.xaxis.set_major_locator(NullLocator())
                    ax.yaxis.set_major_locator(NullLocator())
                else:
                    ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks,
                                                           prune="lower"))
                    ax.yaxis.set_major_locator(NullLocator())
                # Label axes.
                sf = ScalarFormatter(useMathText=use_math_text)
                ax.xaxis.set_major_formatter(sf)
                if i < ndim - 1:
                    if top_ticks:
                        ax.xaxis.set_ticks_position("top")
                        [l.set_rotation(45) for l in ax.get_xticklabels()]
                    else:
                        ax.set_xticklabels([])
                else:
                    [l.set_rotation(45) for l in ax.get_xticklabels()]
                    ax.set_xlabel(labels[i], **label_kwargs)
                    ax.xaxis.set_label_coords(0.5, -0.3)
                # Generate distribution.
                sx = smooth[i]

                # Smooth 1D posterior: use Gaussian-smoothed weighted histogram
                # when `sx` is a float; fall back to standard histogram when
                # `sx` is an integer. This mirrors the smoother used in traceplot
                # and avoids jagged, "hairy" posteriors.
                if isinstance(sx, int_type):
                    n, b, _ = ax.hist(
                        x, bins=sx, weights=weights, color=color,
                        range=np.sort(span[i]), **hist_kwargs
                    )
                    y_max = max(n) if len(n) else 1.0
                else:
                    bins = max(10, int(round(10.0 / sx)))
                    n, b = np.histogram(
                        x, bins=bins, weights=weights, range=np.sort(span[i])
                    )
                    # Apply Gaussian smoothing in bin space with sigma=5 bins
                    # (50% less than before for less aggressive smoothing).
                    # With the chosen bin count this corresponds to a fractional
                    # smoothing width of ~0.5 * `sx` across the parameter span.
                    n = norm_kde(n, 4.0)
                    x0 = 0.5 * (b[1:] + b[:-1])
                    ax.fill_between(x0, n, color=color, **hist_kwargs)
                    y_max = float(n.max()) if n.size else 1.0

                ax.set_ylim([0.0, y_max * 1.05])
                # Plot quantiles.
                if quantiles is not None and len(quantiles) > 0:
                    qs = _quantile(x, quantiles, weights=weights)
                    # for q in qs:
                    #     ax.axvline(q, lw=1, ls="dashed", color=color)
                    if verbose:
                        print("Quantiles:")
                        print(labels[i], [blob for blob in zip(quantiles, qs)])
                # Add truth value(s).
                if truths is not None and truths[i] is not None:
                    try:
                        [ax.axvline(t, color=truth_color, **truth_kwargs)
                         for t in truths[i]]
                    except:
                        ax.axvline(truths[i], color=truth_color, **truth_kwargs)
                # Set titles.
                if show_titles:
                    title = None
                    if title_fmt is not None:
                        ql, qm, qh = _quantile(x, [0.16, 0.5, 0.84], weights=weights)
                        q_minus, q_plus = qm - ql, qh - qm
                        fmt = "{{0:{0}}}".format(title_fmt).format
                        title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
                        title = title.format(fmt(qm), fmt(q_minus), fmt(q_plus))
                        title = "{0} = {1}".format(labels[i], title)
                        ax.set_title(title, **title_kwargs)

                for j, y in enumerate(samples):
                    if np.shape(samples)[0] == 1:
                        ax = axes
                    else:
                        ax = axes[i, j]

                    # Plot the 2-D marginalized posteriors.

                    # Setup axes.
                    if j > i:
                        ax.set_frame_on(False)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        continue
                    elif j == i:
                        continue

                    if max_n_ticks == 0:
                        ax.xaxis.set_major_locator(NullLocator())
                        ax.yaxis.set_major_locator(NullLocator())
                    else:
                        ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks,
                                                               prune="lower"))
                        ax.yaxis.set_major_locator(MaxNLocator(max_n_ticks,
                                                               prune="lower"))
                    # Label axes.
                    sf = ScalarFormatter(useMathText=use_math_text)
                    ax.xaxis.set_major_formatter(sf)
                    ax.yaxis.set_major_formatter(sf)
                    if i < ndim - 1:
                        ax.set_xticklabels([])
                    else:
                        [l.set_rotation(45) for l in ax.get_xticklabels()]
                        ax.set_xlabel(labels[j], **label_kwargs)
                        ax.xaxis.set_label_coords(0.5, -0.3)
                    if j > 0:
                        ax.set_yticklabels([])
                    else:
                        [l.set_rotation(45) for l in ax.get_yticklabels()]
                        ax.set_ylabel(labels[i], **label_kwargs)
                        ax.yaxis.set_label_coords(-0.3, 0.5)
                    # Generate distribution.
                    sy = smooth[j]
                    check_ix = isinstance(sx, int_type)
                    check_iy = isinstance(sy, int_type)
                    if check_ix and check_iy:
                        fill_contours = False
                        plot_contours = False
                    else:
                        fill_contours = True
                        plot_contours = True
                    hist2d_kwargs['fill_contours'] = hist2d_kwargs.get('fill_contours',
                                                                       fill_contours)
                    hist2d_kwargs['plot_contours'] = hist2d_kwargs.get('plot_contours',
                                                                       plot_contours)
                    _hist2d(y, x, ax=ax, span=[span[j], span[i]],
                            weights=weights, color=color, smooth=[sy, sx],
                            **hist2d_kwargs)

                    # Add truth values
                    if truths is not None:
                        if truths[j] is not None:
                            try:
                                [ax.axvline(t, color=truth_color, **truth_kwargs)
                                 for t in truths[j]]
                            except:
                                ax.axvline(truths[j], color=truth_color,
                                           **truth_kwargs)
                        if truths[i] is not None:
                            try:
                                [ax.axhline(t, color=truth_color, **truth_kwargs)
                                 for t in truths[i]]
                            except:
                                ax.axhline(truths[i], color=truth_color,
                                           **truth_kwargs)

            return (fig, axes)

        print('Generating the Posterior Distribution Functions (PDFs) plot')
        _corner_parameters()
        if mds_orig < 2:
            _posteriors_gas_to_vmr(prefix)
            parameters = json.load(open(prefix + 'params.json'))
            a = pymultinest.Analyzer(n_params=len(parameters), outputfiles_basename=prefix, verbose=False)
            data = a.get_data()
            s = a.get_stats()
            i = data[:, 1].argsort()[::-1]
            samples = data[i, 2:]
            weights = data[i, 0]
            loglike = data[i, 1]
            Z = s['global evidence']
            logvol = log(weights) + 0.5 * loglike + Z
            logvol = logvol - logvol.max()  # the (expected) ln(prior volume) associated with each sample

            print("Solution local log-evidence: " + str(s['modes'][0]['local log-evidence']))

            result = dict(samples=samples, weights=weights, logvol=logvol)

            traceplot(result, labels=parameters, show_titles=True)
            plt.savefig(prefix + 'Nest_trace.png', bbox_inches='tight')
            plt.close()

            bound = _plotting_bounds(result)

            if self.param['truths'] is not None:
                tru = np.loadtxt(self.param['truths'])
                cornerplot(result, labels=parameters, show_titles=True, truths=list(tru), span=bound)
            else:
                cornerplot(result, labels=parameters, show_titles=True, span=bound)
            plt.savefig(prefix + 'Nest_posteriors.pdf', bbox_inches='tight')
            plt.close()

            os.system('mv ' + prefix + 'params.json ' + prefix + '_PostProcess.json')
            os.system('mv ' + prefix + 'params_original.json ' + prefix + 'params.json')
            os.system('mv ' + prefix + '.txt ' + prefix + '_PostProcess.txt')
            os.system('mv ' + prefix + 'original.txt ' + prefix + '.txt')
        else:
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
            nest_out = _store_nest_solutions()

            max_ev, max_idx = 0, 0
            for modes in range(0, mds_orig):
                if nest_out['solutions']['solution' + str(modes)]['local_logE'][0] > max_ev:
                    max_ev = nest_out['solutions']['solution' + str(modes)]['local_logE'][0] + 0.0
                    max_idx = modes

            # Store the first solution, the one with maximum local Bayesian evidence
            result = {}
            if self.param['rocky'] and self.param['mod_prior']:
                _posteriors_gas_to_vmr(prefix, modes=max_idx)
            data = np.loadtxt(prefix + 'solution' + str(max_idx) + '.txt')
            i = data[:, 1].argsort()[::-1]
            samples = data[i, 2:]
            weights = data[i, 0]
            loglike = data[i, 1]
            Z = nest_out['global_logE'][0]
            logvol = log(weights) + 0.5 * loglike + Z
            logvol = logvol - logvol.max()  # the (expected) ln(prior volume) associated with each sample

            result['0'] = dict(samples=samples, weights=weights, logvol=logvol)

            to_add = 1
            to_plot = [max_idx]
            for modes in range(0, mds_orig):
                if self.param['filter_multi_solutions']:
                    thrsd = 11.0
                else:
                    thrsd = 1000.0

                if modes == max_idx:
                    print("Solution " + str(modes) + " maximum local log-evidence: " + "{:.2f}".format(nest_out['solutions']['solution' + str(modes)]['local_logE'][0]))
                elif (max_ev - nest_out['solutions']['solution' + str(modes)]['local_logE'][0]) < thrsd:
                    if self.param['rocky'] and self.param['mod_prior']:
                        _posteriors_gas_to_vmr(prefix, modes=modes)
                    data = np.loadtxt(prefix + 'solution' + str(modes) + '.txt')
                    i = data[:, 1].argsort()[::-1]
                    samples = data[i, 2:]
                    weights = data[i, 0]
                    loglike = data[i, 1]
                    Z = nest_out['global_logE'][0]
                    logvol = log(weights) + 0.5 * loglike + Z
                    logvol = logvol - logvol.max()

                    result[str(to_add)] = dict(samples=samples, weights=weights, logvol=logvol)
                    to_add += 1
                    to_plot.append(modes)

                    print("Solution " + str(modes) + ". Local log-evidence: " + "{:.2f}".format(nest_out['solutions']['solution' + str(modes)]['local_logE'][0]))
                else:
                    print("Solution " + str(modes) + " has been skipped. Local log-evidence: " + "{:.2f}".format(nest_out['solutions']['solution' + str(modes)]['local_logE'][0]))

            parameters = json.load(open(prefix + 'params.json'))

            for modes in range(0, to_add):
                traceplot(result[str(modes)], labels=parameters, show_titles=True)
                if len(to_plot) > 1:
                    plt.savefig(prefix + 'Nest_trace (solution' + str(to_plot[modes] + 1) + ').pdf', bbox_inches='tight')
                else:
                    plt.savefig(prefix + 'Nest_trace.pdf', bbox_inches='tight')
                plt.close()

            for modes in range(0, to_add):
                bound = _plotting_bounds(result[str(modes)], modes=None)
                if self.param['truths'] is not None:
                    tru = np.loadtxt(self.param['truths'])
                    cornerplot(result[str(modes)], labels=parameters, show_titles=True, truths=list(tru), color=colors[modes], span=bound)
                else:
                    cornerplot(result[str(modes)], labels=parameters, show_titles=True, color=colors[modes], span=bound)

                if len(to_plot) > 1:
                    plt.savefig(prefix + 'Nest_posteriors (solution' + str(to_plot[modes] + 1) + ').pdf', bbox_inches='tight')
                else:
                    plt.savefig(prefix + 'Nest_posteriors.pdf', bbox_inches='tight')
                plt.close()

            try:
                if figu is not None:
                    pass
            except NameError:
                figu = None

            bound = _plotting_bounds(result, modes=to_add)

            for modes in range(0, to_add):
                if self.param['truths'] is not None:
                    tru = np.loadtxt(self.param['truths'])
                    figu = cornerplot(result[str(modes)], labels=parameters, show_titles=False, truths=list(tru), color=colors[modes], fig=figu, span=bound)
                else:
                    figu = cornerplot(result[str(modes)], labels=parameters, show_titles=False, color=colors[modes], fig=figu, span=bound)

                os.system('mv ' + prefix + 'solution' + str(modes) + '.txt ' + prefix + 'solution' + str(modes) + '_PostProcess.txt')
                os.system('mv ' + prefix + 'solution' + str(modes) + '_original.txt ' + prefix + 'solution' + str(modes) + '.txt')

            plt.savefig(prefix + 'Nest_posteriors.pdf', bbox_inches='tight')
            plt.close()

            os.system('mv ' + prefix + 'params.json ' + prefix + '_PostProcess.json')
            os.system('mv ' + prefix + 'params_original.json ' + prefix + 'params.json')


def plot_posteriors(mnest, prefix, multinest_results, parameters, mds_orig):
    """Plot posterior traces and corner plots (single or multi-mode).

    - Uses the MultiNest outputs at `prefix` to build weighted samples.
    - Produces the same filenames as before for compatibility:
      `Nest_trace.pdf`, `Nest_posteriors.pdf`, and per-solution files when multimodal.
    - Keeps behavior for gas parameter conversion to VMR when applicable.
    - Improves visuals slightly and avoids redundant I/O.
    """
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import pymultinest
    from numpy import log
    from scipy.stats import gaussian_kde
    from scipy.ndimage import gaussian_filter as norm_kde
    from skbio.stats.composition import clr_inv
    from astropy import constants as const

    # Light visual polish
    try:
        plt.style.use('seaborn-v0_8-white')
    except Exception:
        pass
    
    def _posteriors_gas_to_vmr(loc_prefix, modes=None):
        """Convert gas posteriors to VMR space and append mean molecular mass.
        Mirrors previous logic to preserve outputs.
        """
        if modes is None:
            os.system('cp ' + loc_prefix + '.txt ' + loc_prefix + 'original.txt')
            a = np.loadtxt(loc_prefix + '.txt')
        else:
            os.system('cp ' + loc_prefix + 'solution' + str(modes) + '.txt ' + loc_prefix + 'solution' + str(modes) + '_original.txt')
            a = np.loadtxt(loc_prefix + 'solution' + str(modes) + '.txt')

        b = np.ones((len(a[:, 0]), len(a[0, :]) + 2))

        if mnest.param['fit_p0'] and mnest.param['gas_par_space'] != 'partial_pressure':
            b[:, 0:6] = a[:, 0:6] + 0.0
            z = 6
        elif mnest.param['gas_par_space'] == 'partial_pressure':
            b[:, 0:2] = a[:, 0:2] + 0.0
            b[:, 3:6] = a[:, 2:5] + 0.0
            z = 5
        else:
            b[:, 0:5] = a[:, 0:5] + 0.0
            z = 5

        if not mnest.param['fit_wtr_cld']:
            z -= 3
        if mnest.param['fit_amm_cld']:
            b[:, 0:z + 3] = a[:, 0:z + 3] + 0.0
            z += 3

        volume_mixing_ratio = {}
        if mnest.param['gas_par_space'] in ('centered_log_ratio', 'clr'):
            c_l_r = np.array(a[:, z:z + len(mnest.param['fit_molecules'])])
            c_l_r = np.concatenate((c_l_r, np.array([-np.sum(c_l_r, axis=1)]).T), axis=1)
            v_m_r = clr_inv(c_l_r)
            for i, mol in enumerate(mnest.param['fit_molecules']):
                volume_mixing_ratio[mol] = v_m_r[:, i]
            volume_mixing_ratio[mnest.param['gas_fill']] = v_m_r[:, -1]
        elif mnest.param['gas_par_space'] in ('volume_mixing_ratio', 'vmr'):
            volume_mixing_ratio[mnest.param['gas_fill']] = np.ones(len(a[:, 0]))
            for i, mol in enumerate(mnest.param['fit_molecules']):
                vmr_i = 10.0 ** np.array(a[:, z + i])
                volume_mixing_ratio[mol] = vmr_i
                volume_mixing_ratio[mnest.param['gas_fill']] -= vmr_i
        elif mnest.param['gas_par_space'] == 'partial_pressure':
            b[:, 2] = np.sum(10.0 ** np.array(a[:, z:z + len(mnest.param['fit_molecules'])]), axis=1)
            for i, mol in enumerate(mnest.param['fit_molecules']):
                volume_mixing_ratio[mol] = (10.0 ** np.array(a[:, z + i])) / b[:, 2]
            b[:, 2] = np.log10(b[:, 2])

        mmm = np.zeros(len(a[:, 0]))
        for mol in volume_mixing_ratio.keys():
            mmm += volume_mixing_ratio[mol] * mnest.param['mm'][mol]

        if mnest.param['gas_par_space'] != 'partial_pressure':
            for i, mol in enumerate(mnest.param['fit_molecules']):
                b[:, z + i] = np.log10(volume_mixing_ratio[mol])
            if mnest.param['gas_fill'] is not None:
                b[:, z + i + 1] = np.log10(volume_mixing_ratio[mnest.param['gas_fill']])
        else:
            for i, mol in enumerate(mnest.param['fit_molecules']):
                b[:, (z + 1) + i] = np.log10(volume_mixing_ratio[mol])

        b[:, z + i + 2:-1] = a[:, z + i + 1:] + 0.0

        locate_mp_rp = 4 if mnest.param['fit_p_size'] else 3
        if mnest.param['rocky'] and mnest.param['fit_Mp']:
            b[:, -locate_mp_rp] *= (const.M_jup.value / const.M_earth.value)
        if mnest.param['rocky'] and mnest.param['fit_Rp']:
            b[:, -(locate_mp_rp - 1)] *= (const.R_jup.value / const.R_earth.value)

        b[:, -1] = np.array(mmm) + 0.0

        if modes is None:
            np.savetxt(loc_prefix + '.txt', b)
        else:
            np.savetxt(loc_prefix + 'solution' + str(modes) + '.txt', b)

    def _weighted_quantiles(x, qs, w=None):
        x = np.asarray(x)
        qs = np.atleast_1d(qs)
        if w is None:
            return np.percentile(x, list(100.0 * qs))
        w = np.asarray(w)
        sorter = np.argsort(x)
        x, w = x[sorter], w[sorter]
        cw = np.cumsum(w)
        cw /= cw[-1]
        return np.interp(qs, cw, x)

    def _bounds(samples, weights):
        span = 0.999999426697
        lo, hi = [], []
        for i in range(samples.shape[1]):
            # Drop non-finite sample/weight pairs to avoid propagating NaNs into the bounds
            finite = np.isfinite(samples[:, i]) & np.isfinite(weights)
            if not np.any(finite):
                lo.append(0.0)
                hi.append(0.0)
                continue
            col = samples[finite, i]
            w = weights[finite]
            q = [0.5 - 0.5 * span, 0.5 + 0.5 * span]
            if np.sum(w) <= 0:
                v = _weighted_quantiles(col, q, w=None)
            else:
                v = _weighted_quantiles(col, q, w=w)
            lo.append(v[0])
            hi.append(v[1])
        return list(zip(lo, hi))

    def _traceplot(samples, weights, labels, out_path):
        npar = samples.shape[1]
        fig, axes = plt.subplots(npar, 1, figsize=(8, max(2, 1.0 * npar)), dpi=120, sharex=True)
        if npar == 1:
            axes = [axes]
        x = np.arange(samples.shape[0])
        c = (weights - weights.min()) / (weights.max() - weights.min() + 1e-12)
        for i, ax in enumerate(axes):
            ax.scatter(x, samples[:, i], c=c, s=4, cmap='plasma', alpha=0.75)
            ax.set_ylabel(labels[i])
        axes[-1].set_xlabel('Sample index')
        fig.tight_layout()
        plt.savefig(out_path, bbox_inches='tight')
        plt.close(fig)

    def _corner(samples, weights, labels, bounds, truths=None, color='#404784', fig=None):
        npar = samples.shape[1]
        if fig is None:
            fig, axes = plt.subplots(npar, npar, figsize=(2.2 * npar, 2.2 * npar), dpi=130)
        else:
            axes = fig.axes
            axes = np.asarray(axes).reshape(npar, npar)

        for j in range(npar):
            for i in range(npar):
                ax = axes[j, i]
                if j < i:
                    ax.axis('off')
                    continue
                if j == i:
                    lo, hi = bounds[i]
                    mask = (samples[:, i] >= lo) & (samples[:, i] <= hi)
                    grid = np.linspace(lo, hi, 300)
                    try:
                        kde = gaussian_kde(samples[mask, i], weights=weights[mask])
                        dens = kde(grid)
                        ax.plot(grid, dens, color=color, lw=1.6)
                        # quantiles
                        qvals = _weighted_quantiles(samples[:, i], [0.16, 0.5, 0.84], w=weights)
                        for q in qvals:
                            ax.axvline(q, color=color, alpha=0.5, ls='--', lw=1.0)
                        ax.set_yticks([])
                    except Exception:
                        ax.hist(samples[:, i], bins=40, weights=weights, color=color, alpha=0.6)
                        ax.set_yticks([])
                    ax.set_xlim(lo, hi)
                    ax.set_xlabel(labels[i])
                    if truths is not None and truths[i] is not None:
                        try:
                            for t in truths[i]:
                                ax.axvline(t, color='red', lw=1.2)
                        except Exception:
                            ax.axvline(truths[i], color='red', lw=1.2)
                else:
                    # j > i : 2D
                    xlo, xhi = bounds[i]
                    ylo, yhi = bounds[j]
                    xi = samples[:, i]
                    yi = samples[:, j]
                    m = (xi >= xlo) & (xi <= xhi) & (yi >= ylo) & (yi <= yhi)
                    xi, yi, wi = xi[m], yi[m], weights[m]
                    if len(xi) > 10:
                        xx = np.linspace(xlo, xhi, 120)
                        yy = np.linspace(ylo, yhi, 120)
                        xv, yv = np.meshgrid(xx, yy)
                        try:
                            kde = gaussian_kde(np.vstack([xi, yi]), weights=wi)
                            dens = kde(np.vstack([xv.ravel(), yv.ravel()])).reshape(xv.shape)
                            dens = norm_kde(dens, sigma=1.0)
                            cs = ax.contourf(xx, yy, dens, levels=10, cmap='Blues', alpha=0.85)
                            ax.contour(xx, yy, dens, levels=5, colors=[color], linewidths=0.8)
                        except Exception:
                            ax.hist2d(xi, yi, bins=40, weights=wi, cmap='Blues')
                    else:
                        ax.scatter(xi, yi, s=2, color=color, alpha=0.6)
                    ax.set_xlim(xlo, xhi)
                    ax.set_ylim(ylo, yhi)
                    if j == npar - 1:
                        ax.set_xlabel(labels[i])
                    if i == 0:
                        ax.set_ylabel(labels[j])
                    if truths is not None:
                        if truths[i] is not None:
                            try:
                                for t in truths[i]:
                                    ax.axvline(t, color='red', lw=0.8)
                            except Exception:
                                ax.axvline(truths[i], color='red', lw=0.8)
                        if truths[j] is not None:
                            try:
                                for t in truths[j]:
                                    ax.axhline(t, color='red', lw=0.8)
                            except Exception:
                                ax.axhline(truths[j], color='red', lw=0.8)
        fig.tight_layout()
        return fig

    def _corner_parameters():
            if os.path.isfile(prefix + 'params_original.json'):
                pass
            else:
                os.system('mv ' + prefix + 'params.json ' + prefix + 'params_original.json')

            par = []
            if mnest.param['fit_p0'] and mnest.param['gas_par_space'] != 'partial_pressure':
                par.append("Log(P$_0$ [Pa])")
            elif not mnest.param['fit_p0'] and mnest.param['gas_par_space'] == 'partial_pressure':
                par.append("Log(P$_0$ [Pa]) (derived)")
            if mnest.param['fit_wtr_cld']:
                par.append("Log(P$_{top, H_2O}$ [Pa])")
                par.append("Log(D$_{H_2O}$ [Pa])")
                par.append("Log(CR$_{H_2O}$)")
            if mnest.param['fit_amm_cld']:
                par.append("Log(P$_{top, NH_3}$ [Pa])")
                par.append("Log(D$_{NH_3}$ [Pa])")
                par.append("Log(CR$_{NH_3}$)")
            for mol in mnest.param['fit_molecules']:
                par.append(mnest.param['formatted_labels'][mol])
            if mnest.param['gas_fill'] is not None:
                if mnest.param['rocky']:
                    par.append(mnest.param['formatted_labels'][mnest.param['gas_fill']] + " (derived)")
                else:
                    par.append("Log(H$_2$ + He) (derived)")
            if mnest.param['fit_ag']:
                if mnest.param['surface_albedo_parameters'] == int(1):
                    par.append("$a_{surf}$")
                elif mnest.param['surface_albedo_parameters'] == int(3):
                    par.append("$a_{surf, 1}$")
                    par.append("$a_{surf, 2}$")
                    par.append("$\lambda_{surf, 1}$ [$\mu$m]")
                elif mnest.param['surface_albedo_parameters'] == int(5):
                    par.append("$a_{surf, 1}$")
                    par.append("$a_{surf, 2}$")
                    par.append("$a_{surf, 3}$")
                    par.append("$\lambda_{surf, 1}$ [$\mu$m]")
                    par.append("$\lambda_{surf, 2}$ [$\mu$m]")
            if mnest.param['fit_cld_frac']:
                par.append("Log(cld frac)")
            if mnest.param['fit_g']:
                par.append("Log(g [m/s$^2$])")
            if mnest.param['fit_Mp']:
                if mnest.param['rocky']:
                    par.append("M$_p$ [M$_\oplus$]")
                else:
                    par.append("M$_p$ [M$_J$]")
            if mnest.param['fit_Rp']:
                if mnest.param['rocky']:
                    par.append("R$_p$ [R$_\oplus$]")
                else:
                    par.append("R$_p$ [R$_{Jup}$]")
            if mnest.param['fit_p_size'] and mnest.param['p_size_type'] == 'constant':
                par.append("Log(P$_{size}$ [$\mu$m])")
            elif mnest.param['fit_p_size'] and mnest.param['p_size_type'] == 'factor':
                par.append("Log(P$_{size, fctr})$")
            par.append("$\mu$ (derived)")
            json.dump(par, open(prefix + 'params.json', 'w'))

    def _quantile(x, q, weights=None):
            """
            Compute (weighted) quantiles from an input set of samples.
            Parameters
            ----------
            x : `~numpy.ndarray` with shape (nsamps,)
                Input samples.
            q : `~numpy.ndarray` with shape (nquantiles,)
               The list of quantiles to compute from `[0., 1.]`.
            weights : `~numpy.ndarray` with shape (nsamps,), optional
                The associated weight from each sample.
            Returns
            -------
            quantiles : `~numpy.ndarray` with shape (nquantiles,)
                The weighted sample quantiles computed at `q`.
            """

            # Initial check.
            x = np.atleast_1d(x)
            q = np.atleast_1d(q)

            # Quantile check.
            if np.any(q < 0.0) or np.any(q > 1.0):
                raise ValueError("Quantiles must be between 0. and 1.")

            if weights is None:
                # If no weights provided, this simply calls `np.percentile`.
                return np.percentile(x, list(100.0 * q))
            else:
                # If weights are provided, compute the weighted quantiles.
                weights = np.atleast_1d(weights)
                if len(x) != len(weights):
                    raise ValueError("Dimension mismatch: len(weights) != len(x).")
                idx = np.argsort(x)  # sort samples
                sw = weights[idx]  # sort weights
                cdf = np.cumsum(sw)[:-1]  # compute CDF
                cdf /= cdf[-1]  # normalize CDF
                cdf = np.append(0, cdf)  # ensure proper span
                quantiles = np.interp(q, cdf, x[idx]).tolist()
                return quantiles
    
    def _store_nest_solutions():
            NEST_out = {'solutions': {}}
            data = np.loadtxt(prefix + '.txt')
            NEST_stats = multinest_results.get_stats()
            NEST_out['NEST_stats'] = NEST_stats
            NEST_out['global_logE'] = (NEST_out['NEST_stats']['global evidence'], NEST_out['NEST_stats']['global evidence error'])

            modes = []
            modes_weights = []
            modes_loglike = []
            chains = []
            chains_weights = []
            chains_loglike = []

            if mnest.param['multimodal'] and mds_orig > 1:
                # separate modes. get individual samples for each mode
                # get parameter values and sample probability (=weight) for each mode
                with open(prefix + 'post_separate.dat') as f:
                    lines = f.readlines()
                    for idx, line in enumerate(lines):
                        if idx > 2:  # skip the first two lines
                            if lines[idx - 1] == '\n' and lines[idx - 2] == '\n':
                                modes.append(chains)
                                modes_weights.append(chains_weights)
                                modes_loglike.append(chains_loglike)
                                chains = []
                                chains_weights = []
                                chains_loglike = []
                        chain = [float(x) for x in line.split()[2:]]
                        if len(chain) > 0:
                            chains.append(chain)
                            chains_weights.append(float(line.split()[0]))
                            chains_loglike.append(float(line.split()[1]))
                    modes.append(chains)
                    modes_weights.append(chains_weights)
                    modes_loglike.append(chains_loglike)
                modes_array = []
                for mode in modes:
                    mode_array = np.zeros((len(mode), len(mode[0])))
                    for idx, line in enumerate(mode):
                        mode_array[idx, :] = line
                    modes_array.append(mode_array)
            else:
                # not running in multimode. Get chains directly from file prefix.txt
                modes_array = [data[:, 2:]]
                chains_weights = [data[:, 0]]
                modes_weights.append(chains_weights[0])
                chains_loglike = [data[:, 1]]
                modes_loglike.append(chains_loglike[0])
                modes = [0]

            for nmode in range(len(modes)):
                mydict = {'type': 'nest',
                          'local_logE': (NEST_out['NEST_stats']['modes'][nmode]['local log-evidence'], NEST_out['NEST_stats']['modes'][nmode]['local log-evidence error']),
                          'weights': np.asarray(modes_weights[nmode]),
                          'loglike': np.asarray(modes_loglike[nmode]),
                          'tracedata': modes_array[nmode],
                          'fit_params': {}}

                for idx, param_name in enumerate(parameters):
                    trace = modes_array[nmode][:, idx]
                    q_16, q_50, q_84 = _quantile(trace, [0.16, 0.5, 0.84], weights=np.asarray(modes_weights[nmode]))
                    mydict['fit_params'][param_name] = {
                        'value': q_50,
                        'sigma_m': q_50 - q_16,
                        'sigma_p': q_84 - q_50,
                        'nest_map': NEST_stats['modes'][nmode]['maximum a posterior'][idx],
                        'mean': NEST_stats['modes'][nmode]['mean'][idx],
                        'nest_sigma': NEST_stats['modes'][nmode]['sigma'][idx],
                        'trace': trace,
                    }

                NEST_out['solutions']['solution{}'.format(nmode)] = mydict

            if len(NEST_out['solutions']) > 1:
                for i in range(len(NEST_out['solutions'])):
                    fl = np.ones((len(NEST_out['solutions']['solution' + str(i)]['weights']), len(NEST_out['solutions']['solution' + str(i)]['tracedata'][0, :]) + 2))
                    fl[:, 0] = NEST_out['solutions']['solution' + str(i)]['weights']
                    fl[:, 1] = NEST_out['solutions']['solution' + str(i)]['loglike']
                    fl[:, 2:] = NEST_out['solutions']['solution' + str(i)]['tracedata']
                    np.savetxt(prefix + 'solution' + str(i) + '.txt', fl)

            return NEST_out
    # Enhance corner: show median and 1-sigma above 1D plots
    def _annotate_1d_stats(ax, data, weights, fmt='{:g} [−{:g}, +{:g}]'):
        q16, q50, q84 = _weighted_quantiles(data, [0.16, 0.5, 0.84], w=weights)
        lo = q50 - q16
        hi = q84 - q50
        txt = fmt.format(np.round(q50, 4), np.round(lo, 4), np.round(hi, 4))
        ax.text(0.5, 1.02, txt, transform=ax.transAxes, ha='center', va='bottom', fontsize=8)

    # 2D contours use sigma-credible regions at 0.5, 1.0, 2.0, 3.0σ
    # Patch _corner to add stats text and keep speed
    def _corner(samples, weights, labels, bounds, truths=None, color='#404784', fig=None):
        npar = samples.shape[1]
        if fig is None:
            fig, axes = plt.subplots(npar, npar, figsize=(2.2 * npar, 2.2 * npar), dpi=130)
        else:
            axes = fig.axes
            axes = np.asarray(axes).reshape(npar, npar)

        for j in range(npar):
            for i in range(npar):
                ax = axes[j, i]
                if j < i:
                    ax.axis('off')
                    continue
                if j == i:
                    lo, hi = bounds[i]
                    mask = (samples[:, i] >= lo) & (samples[:, i] <= hi)
                    grid = np.linspace(lo, hi, 300)
                    try:
                        # Slightly reduce Scott's factor to sharpen the 1D PDF
                        bw = (lambda s: s.scotts_factor() * 0.25)
                        kde = gaussian_kde(samples[mask, i], weights=weights[mask], bw_method=bw)
                        dens = kde(grid)
                        ax.plot(grid, dens, color=color, lw=1.6)
                        qvals = _weighted_quantiles(samples[:, i], [0.16, 0.5, 0.84], w=weights)
                        for q in qvals:
                            ax.axvline(q, color=color, alpha=0.5, ls='--', lw=1.0)
                        ax.set_yticks([])
                    except Exception:
                        ax.hist(samples[:, i], bins=40, weights=weights, color=color, alpha=0.6)
                        ax.set_yticks([])
                    ax.set_xlim(lo, hi)
                    ax.set_xlabel(labels[i])
                    _annotate_1d_stats(ax, samples[:, i], weights)
                    if truths is not None and truths[i] is not None:
                        try:
                            for t in truths[i]:
                                ax.axvline(t, color='red', lw=1.2)
                        except Exception:
                            ax.axvline(truths[i], color='red', lw=1.2)
                else:
                    xlo, xhi = bounds[i]
                    ylo, yhi = bounds[j]
                    xi = samples[:, i]
                    yi = samples[:, j]
                    m = (xi >= xlo) & (xi <= xhi) & (yi >= ylo) & (yi <= yhi)
                    xi, yi, wi = xi[m], yi[m], weights[m]
                    if len(xi) > 10:
                        xx = np.linspace(xlo, xhi, 120)
                        yy = np.linspace(ylo, yhi, 120)
                        xv, yv = np.meshgrid(xx, yy)
                        try:
                            kde = gaussian_kde(np.vstack([xi, yi]), weights=wi)
                            dens = kde(np.vstack([xv.ravel(), yv.ravel()])).reshape(xv.shape)
                            dens = norm_kde(dens, sigma=1.0)
                            # Compute HPD thresholds corresponding to sigma levels
                            # For a 2D Gaussian, mass within r=sigma is 1 - exp(-sigma^2/2)
                            sigmas = np.array([0.5, 1.0, 2.0, 3.0])
                            probs = 1.0 - np.exp(-0.5 * sigmas * sigmas)
                            flat = dens.ravel()
                            order = np.argsort(flat)[::-1]
                            cdf = np.cumsum(flat[order])
                            cdf /= (cdf[-1] + 1e-300)
                            thr = np.interp(probs, cdf, flat[order])
                            # Draw a light filled background for readability
                            ax.contourf(xx, yy, dens, levels=12, cmap='Blues', alpha=0.75)
                            # Overlay sigma-level contour lines (HPD)
                            ax.contour(xx, yy, dens, levels=np.sort(thr), colors=[color], linewidths=1.0)
                        except Exception:
                            ax.hist2d(xi, yi, bins=40, weights=wi, cmap='Blues')
                    else:
                        ax.scatter(xi, yi, s=2, color=color, alpha=0.6)
                    ax.set_xlim(xlo, xhi)
                    ax.set_ylim(ylo, yhi)
                    if j == npar - 1:
                        ax.set_xlabel(labels[i])
                    if i == 0:
                        ax.set_ylabel(labels[j])
                    if truths is not None:
                        if truths[i] is not None:
                            try:
                                for t in truths[i]:
                                    ax.axvline(t, color='red', lw=0.8)
                            except Exception:
                                ax.axvline(truths[i], color='red', lw=0.8)
                        if truths[j] is not None:
                            try:
                                for t in truths[j]:
                                    ax.axhline(t, color='red', lw=0.8)
                            except Exception:
                                ax.axhline(truths[j], color='red', lw=0.8)
        fig.tight_layout()
        return fig
    
    def _plot_1d_posteriors(sample_sets, weight_sets, labels, bounds, outfile, colors, truths=None, legend_labels=None):
        """Create grid of 1D posterior PDFs (matching the corner diagonal panels)."""
        from matplotlib.lines import Line2D
        sample_sets = sample_sets if isinstance(sample_sets, (list, tuple)) else [sample_sets]
        weight_sets = weight_sets if isinstance(weight_sets, (list, tuple)) else [weight_sets]
        sample_sets = [np.asarray(s) for s in sample_sets]
        weight_sets = [np.asarray(w) for w in weight_sets]
        npar = sample_sets[0].shape[1]

        def _grid_shape(npars):
            if npars <= 0:
                return 1, 1
            cols = int(np.ceil(np.sqrt(2 * npars)))
            cols = max(cols, 2)
            rows = int(np.ceil(npars / cols))
            while rows * 2 > cols:
                cols = rows * 2
                rows = int(np.ceil(npars / cols))
            return rows, cols

        rows, cols = _grid_shape(npar)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.8, rows * 2.2), dpi=130)
        axes = np.atleast_1d(axes).reshape(rows, cols)
        flat_axes = axes.ravel()

        for idx, ax in enumerate(flat_axes):
            if idx >= npar:
                ax.axis('off')
                continue
            lo, hi = bounds[idx]
            grid = np.linspace(lo, hi, 320)
            ymax = 0.0
            for j, (samples, weights) in enumerate(zip(sample_sets, weight_sets)):
                color = colors[j % len(colors)]
                vec = samples[:, idx]
                msk = (vec >= lo) & (vec <= hi)
                try:
                    if msk.sum() > 4:
                        kde = gaussian_kde(vec[msk], weights=weights[msk],
                                           bw_method=lambda s: s.scotts_factor() * 0.25)
                        dens = kde(grid)
                        ax.plot(grid, dens, color=color, lw=1.6, alpha=0.9)
                        ymax = max(ymax, float(dens.max()) if dens.size else ymax)
                    else:
                        raise RuntimeError("Too few points for KDE")
                except Exception:
                    hist, edges = np.histogram(vec, bins=40, weights=weights, range=(lo, hi))
                    centers = 0.5 * (edges[1:] + edges[:-1])
                    ax.plot(centers, hist, color=color, lw=1.2, alpha=0.9, drawstyle='steps-mid')
                    ymax = max(ymax, float(hist.max()) if hist.size else ymax)
            if truths is not None and truths[idx] is not None:
                try:
                    iterator = truths[idx]
                    for t in iterator:
                        ax.axvline(t, color='red', lw=1.0, alpha=0.8)
                except Exception:
                    ax.axvline(truths[idx], color='red', lw=1.0, alpha=0.8)
            ax.set_xlim(lo, hi)
            ax.set_xlabel(labels[idx])
            ax.set_yticks([])
            if ymax > 0.0:
                ax.set_ylim(0.0, ymax * 1.05)
            _annotate_1d_stats(ax, sample_sets[0][:, idx], weight_sets[0])

        if legend_labels and len(sample_sets) > 1:
            handles = [Line2D([0], [0], color=colors[i % len(colors)], lw=1.6, label=legend_labels[i])
                       for i in range(len(sample_sets))]
            fig.legend(handles=handles, loc='upper right', fontsize=8, frameon=False)

        fig.tight_layout()
        fig.savefig(outfile, bbox_inches='tight')
        plt.close(fig)

    def _corner_selected(labels):
        """Return sorted label indices requested for the corner plot, or None."""
        sel_cfg = mnest.param.get('corner_selected_params')
        if not sel_cfg:
            return None

        from collections import defaultdict
        import re

        if isinstance(sel_cfg, str):
            raw_terms = [chunk.strip() for chunk in re.split(r'[;,]', sel_cfg) if chunk.strip()]
        else:
            raw_terms = [str(chunk).strip() for chunk in sel_cfg if str(chunk).strip()]
        if not raw_terms:
            return None

        label_lookup = defaultdict(list)
        for idx, label in enumerate(labels):
            label_lookup[label].append(idx)

        selected = []
        used = set()
        missing = []

        for raw in raw_terms:
            candidates = label_lookup.get(raw, [])
            if not candidates:
                missing.append(raw)
                continue

            chosen = None
            for idx in candidates:
                if idx not in used:
                    chosen = idx
                    break
            if chosen is None and candidates:
                chosen = candidates[0]
            if chosen is not None:
                used.add(chosen)
                selected.append(chosen)

        if missing:
            print(f"corner_selected_params - could not match: {', '.join(missing)}")

        if not selected:
            return None
        return sorted(set(selected))

    print('Generating the Posterior Distribution Functions (PDFs) plot')

    _corner_parameters()
    # Single-mode
    if mds_orig < 2:
        _posteriors_gas_to_vmr(prefix)

        # Read MultiNest data through Analyzer to be robust
        a = pymultinest.Analyzer(n_params=len(parameters), outputfiles_basename=prefix, verbose=False)
        data = a.get_data()
        s = a.get_stats()
        order = data[:, 1].argsort()[::-1]
        samples = data[order, 2:]
        weights = data[order, 0]
        loglike = data[order, 1]
        Z = s.get('global evidence', s.get('global_logE', [0]))
        if isinstance(Z, (list, tuple, np.ndarray)):
            Z = Z[0]
        logvol = log(weights) + 0.5 * loglike + Z
        logvol = logvol - logvol.max()

        print("Solution global log-evidence: " + str(s['modes'][0]['local log-evidence']))

        labels = json.load(open(prefix + 'params.json'))

        # Trace
        _traceplot(samples, weights, labels, prefix + 'Nest_trace.png')
        
        bounds = _bounds(samples, weights)
        truths = None
        if mnest.param.get('truths') is not None:
            try:
                truths = list(np.loadtxt(mnest.param['truths']))
            except Exception:
                truths = None
        if truths is not None:
            truths = list(truths) + [None] * (len(labels) - len(truths))
        selected_idx = _corner_selected(labels)
        corner_labels_all = [labels[i] for i in selected_idx] if selected_idx else labels
        corner_labels = [labels[i] for i in selected_idx] if selected_idx else labels
        if selected_idx:
            corner_samples = samples[:, selected_idx]
            corner_bounds = [bounds[i] for i in selected_idx]
            corner_truths = [truths[i] for i in selected_idx] if truths is not None else None
        else:
            corner_samples = samples
            corner_bounds = bounds
            corner_truths = truths
        fig = _corner(corner_samples, weights, corner_labels, corner_bounds, truths=corner_truths, color='#404784')
        
        if mnest.param.get('corner_selected_params') is None:
            plt.savefig(prefix + 'Nest_posteriors.pdf', bbox_inches='tight')
        else:
            plt.savefig(prefix + 'Nest_selected_posteriors.pdf', bbox_inches='tight')
        plt.close(fig)

        if mnest.param.get('corner_selected_params') is None:
            _plot_1d_posteriors(corner_samples, weights, corner_labels, corner_bounds,
                                prefix + 'Nest_1D_posteriors.pdf', colors=['#404784'],
                                truths=corner_truths)
        else:
            _plot_1d_posteriors(corner_samples, weights, corner_labels, corner_bounds,
                                prefix + 'Nest_selected_1D_posteriors.pdf', colors=['#404784'],
                                truths=corner_truths)

        # Restore modified files (if any)
        if mnest.param['rocky'] and mnest.param['mod_prior']:
            os.system('mv ' + prefix + '.txt ' + prefix + '_PostProcess.txt')
            os.system('mv ' + prefix + 'original.txt ' + prefix + '.txt')
        if os.path.isfile(prefix + 'params_original.json'):
            os.system('mv ' + prefix + 'params.json ' + prefix + '_PostProcess.json')
            os.system('mv ' + prefix + 'params_original.json ' + prefix + 'params.json')

    # Multi-modal
    else:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

        # Load stats.json (as in original) to obtain local evidences
        nest_out = _store_nest_solutions()

        # Pick maximum-evidence mode first
        max_ev, max_idx = -1e99, 0
        for modes in range(0, mds_orig):
            loc = nest_out['solutions']['solution' + str(modes)]['local_logE'][0]
            if loc > max_ev:
                max_ev, max_idx = loc, modes

        result = {}
        to_add = 0
        to_plot = []
        kept_modes = []

        # Build a small helper to read solution file -> samples/weights/logvol
        def _read_solution(midx):
            _posteriors_gas_to_vmr(prefix, modes=midx)
            data = np.loadtxt(prefix + 'solution' + str(midx) + '.txt')
            order = data[:, 1].argsort()[::-1]
            samples = data[order, 2:]
            weights = data[order, 0]
            loglike = data[order, 1]
            Z = nest_out.get('global_logE', [0])[0]
            logvol = log(weights) + 0.5 * loglike + Z
            logvol = logvol - logvol.max()
            return samples, weights, logvol

        # Always include the maximum-evidence solution
        s0, w0, lv0 = _read_solution(max_idx)
        result[str(to_add)] = dict(samples=s0, weights=w0, logvol=lv0)
        to_plot.append(max_idx)
        kept_modes.append(max_idx)
        to_add += 1

        # Add other significant modes
        thresh = 11.0 if mnest.param.get('filter_multi_solutions') else 1000.0
        for modes in range(0, mds_orig):
            if modes == max_idx:
                continue
            local = nest_out['solutions']['solution' + str(modes)]['local_logE'][0]
            if (max_ev - local) < thresh:
                s1, w1, lv1 = _read_solution(modes)
                result[str(to_add)] = dict(samples=s1, weights=w1, logvol=lv1)
                to_plot.append(modes)
                kept_modes.append(modes)
                to_add += 1

        labels = json.load(open(prefix + 'params.json'))
        selected_idx = _corner_selected(labels)

        # Individual traces and corner plots
        for k, midx in enumerate(to_plot):
            _traceplot(result[str(k)]['samples'], result[str(k)]['weights'], labels,
                       prefix + ('Nest_trace (solution' + str(midx + 1) + ').png' if len(to_plot) > 1 else 'Nest_trace.png'))

            bnd = _bounds(result[str(k)]['samples'], result[str(k)]['weights'])
            truths = None
            if mnest.param.get('truths') is not None:
                try:
                    truths = list(np.loadtxt(mnest.param['truths']))
                except Exception:
                    truths = None
            if truths is not None:
                truths = list(truths) + [None] * (len(labels) - len(truths))
            corner_labels = corner_labels_all
            if selected_idx:
                corner_samples = result[str(k)]['samples'][:, selected_idx]
                corner_bounds = [bnd[i] for i in selected_idx]
                corner_truths = [truths[i] for i in selected_idx] if truths is not None else None
            else:
                corner_samples = result[str(k)]['samples']
                corner_bounds = bnd
                corner_truths = truths
            fig = _corner(corner_samples, result[str(k)]['weights'], corner_labels, corner_bounds,
                          truths=corner_truths, color=colors[k])
            outp = prefix + ('Nest_posteriors (solution' + str(midx + 1) + ').pdf' if len(to_plot) > 1 else 'Nest_posteriors.pdf')
            plt.savefig(outp, bbox_inches='tight')
            plt.close(fig)

        # Combined overlay corner plot
        # Determine union bounds across kept modes
        mins, maxs = None, None
        for k in range(to_add):
            b = _bounds(result[str(k)]['samples'], result[str(k)]['weights'])
            lo = np.array([bb[0] for bb in b])
            hi = np.array([bb[1] for bb in b])
            mins = lo if mins is None else np.minimum(mins, lo)
            maxs = hi if maxs is None else np.maximum(maxs, hi)
        union_bounds = list(zip(mins.tolist(), maxs.tolist()))

        fig = None
        truths = None
        if mnest.param.get('truths') is not None:
            try:
                truths = list(np.loadtxt(mnest.param['truths']))
            except Exception:
                truths = None
        if truths is not None:
            truths = list(truths) + [None] * (len(labels) - len(truths))
        overlay_labels = corner_labels_all
        if selected_idx:
            overlay_bounds = [union_bounds[i] for i in selected_idx]
            overlay_truths = [truths[i] for i in selected_idx] if truths is not None else None
        else:
            overlay_bounds = union_bounds
            overlay_truths = truths
        for k in range(to_add):
            overlay_samples = result[str(k)]['samples'][:, selected_idx] if selected_idx else result[str(k)]['samples']
            fig = _corner(overlay_samples, result[str(k)]['weights'], overlay_labels,
                          overlay_bounds, truths=overlay_truths, color=colors[k], fig=fig)

        if mnest.param.get('corner_selected_params') is None:
            plt.savefig(prefix + 'Nest_posteriors.pdf', bbox_inches='tight')
        else:
            plt.savefig(prefix + 'Nest_selected_posteriors.pdf', bbox_inches='tight')
        plt.close()

        sample_sets = [result[str(k)]['samples'] for k in range(to_add)]
        weight_sets = [result[str(k)]['weights'] for k in range(to_add)]
        legend_labels = [f'Solution {to_plot[k] + 1}' for k in range(to_add)] if to_add > 1 else None
        sel_colors = [colors[k % len(colors)] for k in range(to_add)]
        plot_labels = corner_labels_all
        if selected_idx:
            plot_bounds = [union_bounds[i] for i in selected_idx]
            plot_truths = [truths[i] for i in selected_idx] if truths is not None else None
            sample_sets = [s[:, selected_idx] for s in sample_sets]
        else:
            plot_bounds = union_bounds
            plot_truths = truths
        
        if mnest.param.get('corner_selected_params') is None:
            _plot_1d_posteriors(sample_sets, weight_sets, plot_labels, plot_bounds,
                                prefix + 'Nest_1D_posteriors.pdf', colors=sel_colors,
                                truths=plot_truths, legend_labels=legend_labels)
        else:
            _plot_1d_posteriors(sample_sets, weight_sets, plot_labels, plot_bounds,
                                prefix + 'Nest_selected_1D_posteriors.pdf', colors=sel_colors,
                                truths=plot_truths, legend_labels=legend_labels)

        # Restore modified files (if any)
        for modes in kept_modes:
            if mnest.param['rocky'] and mnest.param['mod_prior']:
                os.system('mv ' + prefix + 'solution' + str(modes) + '.txt ' + prefix + 'solution' + str(modes) + '_PostProcess.txt')
                os.system('mv ' + prefix + 'solution' + str(modes) + '_original.txt ' + prefix + 'solution' + str(modes) + '.txt')
        if os.path.isfile(prefix + 'params_original.json'):
            os.system('mv ' + prefix + 'params.json ' + prefix + '_PostProcess.json')
            os.system('mv ' + prefix + 'params_original.json ' + prefix + 'params.json')


def plot_contribution(mnest, cube, solutions=None):
    """Plot per-molecule spectral contributions at R≈500 and export components.

    Parameters
    - mnest: MULTINEST instance (provides `param`, `cube_to_param`, `adjust_for_cld_frac`).
    - cube: parameter cube (array-like)
    - solutions: Optional solution index for filenames

    Behavior
    - Saves per-molecule contribution curves to `contr_comp[/ _{solutions}]/*.dat`.
    - Plots all contributions, the cloud-only curve, and overlays the data with errors.
    - Restores all temporarily modified `mnest.param` fields.
    """
    # Mild aesthetic improvements
    try:
        plt.style.use('seaborn-v0_8-white')
    except Exception:
        pass

    # Output directory handling (single check, avoid repeated os.path operations)
    if solutions is None:
        subdir = 'contr_comp/'
    else:
        subdir = f'contr_comp_{solutions}/'
    out_dir = mnest.param['out_dir'] + subdir
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    # Load R=500 bins once
    new_wl = np.loadtxt(mnest.param['pkg_dir'] + 'forward_mod/Data/wl_bins/bins_02_20_R500.dat')
    new_wl_central = np.mean(new_wl, axis=1)

    # Cache and temporarily override spectrum grid/bins
    is_bins = mnest.param['spectrum']['bins']
    mnest.param['spectrum']['bins'] = False

    if mnest.param['mol_custom_wl']:
        start, stop = 0, len(new_wl_central) - 1
    else:
        start = find_nearest(new_wl_central, min(mnest.param['spectrum']['wl']) - 0.05)
        stop = find_nearest(new_wl_central, max(mnest.param['spectrum']['wl']) + 0.05)

    temp_wl = mnest.param['spectrum']['wl'] + 0.0
    mnest.param['spectrum']['wl'] = new_wl_central[start:stop]
    mnest.param['start_c_wl_grid'] = find_nearest(mnest.param['wl_C_grid'], float(np.min(mnest.param['spectrum']['wl']))) - 35
    mnest.param['stop_c_wl_grid'] = find_nearest(mnest.param['wl_C_grid'], float(np.max(mnest.param['spectrum']['wl']))) + 35

    # Prepare figure
    fig = plt.figure(figsize=(10.5, 5.5), dpi=140)

    # Toggle contribution mode and compute each molecule contribution on the same grid
    mnest.param['contribution'] = True
    for mol in mnest.param['fit_molecules']:
        print('Plotting the contribution of ' + str(mol) + ' : VMR -> ' + str(mnest.param['vmr_' + mol][-1]))
        mnest.param['mol_contr'] = mol
        mod = _instantiate_forward_model(mnest.param)
        alb_wl, alb = mod.run_forward()

        if mnest.param['fit_wtr_cld'] and mnest.param['cld_frac'] != 1.0:
            alb = mnest.adjust_for_cld_frac(alb, cube)
            mnest.cube_to_param(cube)

        wl, model = model_finalizzation(mnest.param, alb_wl, alb,
                                        planet_albedo=mnest.param['albedo_calc'],
                                        fp_over_fs=mnest.param['fp_over_fs'])
        plt.plot(wl, model, linewidth=1.4, label=mol)
        np.savetxt(out_dir + mol + '.dat', np.column_stack([wl, model]))

    # Cloud-only curve (keep contribution=True, remove molecule tag)
    mnest.param['mol_contr'] = None
    mod = _instantiate_forward_model(mnest.param)
    alb_wl, alb = mod.run_forward()
    if mnest.param['fit_wtr_cld'] and mnest.param['cld_frac'] != 1.0:
        alb = mnest.adjust_for_cld_frac(alb, cube)
        mnest.cube_to_param(cube)
    wl, model = model_finalizzation(mnest.param, alb_wl, alb,
                                    planet_albedo=mnest.param['albedo_calc'],
                                    fp_over_fs=mnest.param['fp_over_fs'])
    plt.plot(wl, model, color='black', linestyle='--', linewidth=1.7, alpha=0.9, label='H$_2$O cloud')
    np.savetxt(out_dir + 'H2O_cld.dat', np.column_stack([wl, model]))
    mnest.param['contribution'] = False

    # Restore observed wavelength grid extents for data overlay
    mnest.param['spectrum']['wl'] = temp_wl + 0.0
    mnest.param['start_c_wl_grid'] = find_nearest(mnest.param['wl_C_grid'], mnest.param['min_wl']) - 35
    mnest.param['stop_c_wl_grid'] = find_nearest(mnest.param['wl_C_grid'], mnest.param['max_wl']) + 35

    # Data overlay
    plt.errorbar(
        mnest.param['spectrum']['wl'], mnest.param['spectrum']['Fplanet'],
        yerr=mnest.param['spectrum']['error_p'], linestyle='', linewidth=0.6,
        color='#111111', marker='o', markerfacecolor='#e15759', markeredgecolor='#111111',
        markersize=4.2, capsize=1.8, label='Data')

    # Labels and legend
    plt.xlabel('Wavelength [$\mu$m]')
    if mnest.param['albedo_calc']:
        plt.ylabel('Albedo')
    elif mnest.param['fp_over_fs']:
        plt.ylabel('Contrast Ratio (F$_p$/F$_{\star}$)')
    else:
        plt.ylabel('Planetary flux [W/m$^2$]')

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
    fig.tight_layout()

    # Save figure
    if mnest.param['mol_custom_wl']:
        if solutions is None:
            plt.savefig(mnest.param['out_dir'] + 'Nest_mol_contribution_extended.pdf')
        else:
            plt.savefig(mnest.param['out_dir'] + f'Nest_mol_contribution_extended (solution {solutions}).pdf')
    else:
        if solutions is None:
            plt.savefig(mnest.param['out_dir'] + 'Nest_mol_contribution.pdf')
        else:
            plt.savefig(mnest.param['out_dir'] + f'Nest_mol_contribution (solution {solutions}).pdf')
    plt.close()

    # Restore original bins flag
    if is_bins:
        mnest.param['spectrum']['bins'] = True


def plot_chemistry(param, solutions=None):
    """Plot retrieved atmospheric chemistry profiles and mean molecular mass.

    Parameters
    - param: Parameter dictionary containing atmospheric structure, fitted molecules
      and plotting configuration. Expected keys include (non-exhaustive):
      `fit_molecules`, `P`, `vmr_<mol>`, `rocky`, `P0`, `fit_wtr_cld`, `Pw_top`,
      `cldw_depth`, `gas_fill`, `vmr_He`, `out_dir`, `O3_earth`, `mean_mol_weight`.
    - solutions: Optional integer used to suffix output filenames to distinguish
      between multiple solutions.

    Behavior
    - Creates two figures:
      1) Volume mixing ratio (VMR) profiles vs pressure for each fitted molecule
         (and optional background gases), with markers for surface and cloud
         boundaries when applicable. Saved as `Nest_chemistry[ (solution N)].pdf`.
      2) Mean molecular mass vs pressure, saved as `Nest_MMM[ (solution N)].pdf`.
    - Prints top and bottom VMRs (or stratospheric value for O3 when `O3_earth`).
    - Uses Pa on the primary y-axis and adds a secondary axis in bar.
    """
    # Chemistry plot
    fig, ax = plt.subplots()

    for mol in param['fit_molecules']:
        ax.loglog(param['vmr_' + mol], param['P'], label=mol)
        if mol == 'O3' and param['O3_earth']:
            print(
                str(mol)
                + ' -> Stratosphere: '
                + str(param['vmr_' + mol][np.where(param['vmr_' + mol] > 10.0 ** (-11.0))[0][0]])
                + ', Elsewhere: 0.0'
            )
        else:
            print(
                str(mol)
                + ' -> Top: '
                + str(param['vmr_' + mol][0])
                + ', Bottom: '
                + str(param['vmr_' + mol][-1])
            )

    if param['gas_fill'] is not None:
        if not param['rocky']:
            print('He' + ' -> Top: ' + str(param['vmr_He'][0]) + ', Bottom: ' + str(param['vmr_He'][-1]))
            ax.loglog(param['vmr_He'], param['P'], label='He')
        print(
            str(param['gas_fill'])
            + ' -> Top: '
            + str(param['vmr_' + param['gas_fill']][0])
            + ', Bottom: '
            + str(param['vmr_' + param['gas_fill']][-1])
        )
        ax.loglog(param['vmr_' + param['gas_fill']], param['P'], label=param['gas_fill'])

    ax.set_xlim((1e-18, 1.5))
    ax.set_xlabel('Molecule VMR')
    ax.set_ylabel('Pressure [Pa]')

    def pa_to_bar(y):
        return y / (10.0 ** 5.0)

    def bar_to_pa(y):
        return y * (10.0 ** 5.0)

    if param['fit_wtr_cld']:
        pos_cldw = int(find_nearest(param['P'], param['Pw_top']))

        plt.hlines(
            param['P'][int(find_nearest(param['P'], (param['cldw_depth'] + param['P'][pos_cldw])))],
            ax.get_xlim()[0],
            ax.get_xlim()[1],
            linestyle='--',
            alpha=0.5,
            color='black',
            label='H$_2$O cloud',
        )
        plt.hlines(
            param['P'][int(find_nearest(param['P'], param['Pw_top']))],
            ax.get_xlim()[0],
            ax.get_xlim()[1],
            linestyle='--',
            alpha=0.5,
            color='black',
        )
    if param['rocky']:
        plt.hlines(
            param['P'][int(find_nearest(param['P'], param['P0']))],
            ax.get_xlim()[0],
            ax.get_xlim()[1],
            linestyle='-',
            color='black',
            alpha=0.75,
            label='Surface',
        )

    ax.yaxis.set_ticks(10.0 ** np.arange(1.0, 12.1, 1))
    if sys.version[0] == '3':
        secax_y = ax.secondary_yaxis('right', functions=(pa_to_bar, bar_to_pa))
        secax_y.set_ylabel('Pressure [bar]')

    if param['rocky']:
        ax.set_ylim((1.0, param['P0'] + (0.1 * param['P0'])))
    else:
        bottom = 5 * param['P'][int(find_nearest(param['P'], (param['cldw_depth'] + param['P'][pos_cldw])))]
        ax.set_ylim([1.0, bottom])
    plt.gca().invert_yaxis()

    ax.legend(loc='upper left', framealpha=0)
    if solutions is None:
        plt.savefig(param['out_dir'] + 'Nest_chemistry.pdf')
    else:
        plt.savefig(param['out_dir'] + 'Nest_chemistry (solution ' + str(solutions) + ').pdf')
    plt.close()

    # Mean Molecular Mass plot
    fig, ax = plt.subplots()
    ax.semilogy(param['mean_mol_weight'], param['P'], label='Mean Molecular Mass')
    ax.set_xlabel('Mean molecular weight')
    ax.set_ylabel('Pressure [Pa]')

    ax.set_xlim((ax.get_xlim()[0], ax.get_xlim()[1]))

    if param['fit_wtr_cld']:
        plt.hlines(
            param['P'][int(find_nearest(param['P'], (param['cldw_depth'] + param['P'][pos_cldw])))],
            ax.get_xlim()[0],
            ax.get_xlim()[1],
            linestyle='--',
            alpha=0.5,
            color='black',
            label='H$_2$O cloud',
        )
        plt.hlines(
            param['P'][int(find_nearest(param['P'], param['Pw_top']))],
            ax.get_xlim()[0],
            ax.get_xlim()[1],
            linestyle='--',
            alpha=0.5,
            color='black',
        )
    if param['rocky']:
        plt.hlines(
            param['P'][int(find_nearest(param['P'], param['P0']))],
            ax.get_xlim()[0],
            ax.get_xlim()[1],
            linestyle='-',
            color='black',
            alpha=0.75,
            label='Surface',
        )

    ax.yaxis.set_ticks(10.0 ** np.arange(1.0, 12.1, 1))
    if sys.version[0] == '3':
        secax_y = ax.secondary_yaxis('right', functions=(pa_to_bar, bar_to_pa))
        secax_y.set_ylabel('Pressure [bar]')

    if param['rocky']:
        ax.set_ylim((1.0, param['P0'] + (0.1 * param['P0'])))
    else:
        bottom = 5 * param['P'][int(find_nearest(param['P'], (param['cldw_depth'] + param['P'][pos_cldw])))]
        ax.set_ylim([1.0, bottom])
    plt.gca().invert_yaxis()

    ax.legend(framealpha=0)
    if solutions is None:
        plt.savefig(param['out_dir'] + 'Nest_MMM.pdf')
    else:
        plt.savefig(param['out_dir'] + 'Nest_MMM (solution ' + str(solutions) + ').pdf')
    plt.close()


def plot_surface_albedo(param, solutions=None):
    """Plot the retrieved piecewise-constant surface albedo function.

    Parameters
    - param: Parameter dictionary containing plotting configuration and the
      albedo step parameters. Expected keys include:
      `surface_albedo_parameters` (int, either 3 or 5), `Ag_x1`, `Ag_x2`
      (optional when 3-parameter), `Ag1`, `Ag2`, `Ag3` (optional when
      3-parameter), `min_wl`, `max_wl`, and `out_dir`.
    - solutions: Optional integer used to suffix the output filename.

    Behavior
    - Builds a wavelength array in [`min_wl`, `max_wl`] and renders a step
      function with one transition (`Ag_x1`) for the 3-parameter case, or two
      transitions (`Ag_x1`, `Ag_x2`) for the 5-parameter case.
    - Saves the figure to `Nest_surface_albedo.pdf` (or with a solution suffix).
    """
    # Define parameters for the step function
    x1 = param['Ag_x1'] + 0.0
    a1, a2 = param['Ag1'] + 0.0, param['Ag2'] + 0.0
    if param['surface_albedo_parameters'] == int(5):
        x2 = param['Ag_x2'] + 0.0  # wavelength cutoffs in microns
        a3 = param['Ag3'] + 0.0  # albedo values for each region

    # Create data for plotting
    wavelength = np.linspace(param['min_wl'], param['max_wl'], 1000)  # x-axis: wavelength in microns

    # Create the step function
    step_albedo = np.zeros_like(wavelength)
    step_albedo[wavelength < x1] = a1
    if param['surface_albedo_parameters'] == int(3):
        step_albedo[wavelength >= x1] = a2
    elif param['surface_albedo_parameters'] == int(5):
        step_albedo[(wavelength >= x1) & (wavelength < x2)] = a2
        step_albedo[wavelength >= x2] = a3

    # Create a high-quality figure
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    # Plot with enhanced styling
    ax.plot(wavelength, step_albedo, linewidth=3, color='#ff7f0e', label='Retrieved surface albedo function')

    # Fill areas under curves for visual enhancement
    ax.fill_between(wavelength, 0, step_albedo, alpha=0.3, color='#ff7f0e')

    # Add vertical lines at transition points
    ax.axvline(x=x1, color='#9467bd', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Transition $\\lambda_1$: {np.round(x1,2)} $\\mu$m')
    if param['surface_albedo_parameters'] == int(5):
        ax.axvline(x=x2, color='#d62728', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Transition $\\lambda_2$: {np.round(x2,2)} $\\mu$m')

    # Annotations for step values
    ax.annotate(f'a$_1$ = {np.round(a1,2)}', xy=((param['min_wl'] + x1) / 2, a1), xytext=((param['min_wl'] + x1) / 2, a1 + 0.04),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=12)
    if param['surface_albedo_parameters'] == int(3):
        ax.annotate(f'a$_2$ = {np.round(a2, 2)}', xy=((x1 + param['max_wl']) / 2, a2), xytext=((x1 + param['max_wl']) / 2, a2 + 0.04),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                    fontsize=12)
    elif param['surface_albedo_parameters'] == int(5):
        ax.annotate(f'a$_2$ = {np.round(a2,2)}', xy=((x1 + x2) / 2, a2), xytext=((x1 + x2) / 2, a2 + 0.04),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                    fontsize=12)
        ax.annotate(f'a$_3$ = {np.round(a3,2)}', xy=((x2 + param['max_wl']) / 2, a3), xytext=((x2 + param['max_wl']) / 2, a3 + 0.04),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                    fontsize=12)

    # Enhance the axes and labels
    ax.set_xlabel('Wavelength [$\\mu$m]', fontsize=14, fontweight='bold')
    ax.set_ylabel('Surface Albedo', fontsize=14, fontweight='bold')

    # Set axis limits with a bit of padding
    if param['surface_albedo_parameters'] == int(3):
        ax.set_ylim(-0.02, np.max([a1, a2]) + 0.1)
    elif param['surface_albedo_parameters'] == int(5):
        ax.set_ylim(-0.02, np.max([a1, a2, a3]) + 0.1)

    # Add a grid for better readability
    ax.grid(True, linestyle='--', alpha=0.3)

    # Format tick labels
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    # Customize tick parameters
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Adjust layout
    plt.tight_layout()

    if solutions is None:
        plt.savefig(param['out_dir'] + 'Nest_surface_albedo.pdf')
    else:
        plt.savefig(param['out_dir'] + 'Nest_surface_albedo (solution ' + str(solutions) + ').pdf')

    plt.close()
