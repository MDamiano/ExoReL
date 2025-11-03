import os
import sys
import copy
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from astropy import constants as const

from .__utils import find_nearest, model_finalizzation, temp_profile, reso_range
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
            new_wl = reso_range(mnest.param['min_wl'] - 0.05, mnest.param['max_wl'] + 0.05, 500, bins=True)
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

        if not mnest.param['fit_wtr_cld'] or mnest.param['PT_profile_type'] == 'parametric':
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
            b[:, 2] = np.sum(10.0 ** np.array(a[:, z:z + len(mnest.param['fit_molecules'])]), axis=1)  # PDF of P0 (surface pressure)
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
        if mnest.param['fit_g']:
            b[:, -(locate_mp_rp - 2)] = 10. ** (b[:, -(locate_mp_rp - 2)] - 2.0)

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
            if mnest.param['fit_T']:
                if mnest.param['PT_profile_type'] == 'isothermal':
                    par.append("T$_p$")
                elif mnest.param['PT_profile_type'] == 'parametric':
                    par.append("$\kappa_{th}$")
                    par.append("$\gamma$")
                    par.append("$\\beta$")
                    if mnest.param['fit_Tint']:
                        par.append("T$_{int}$")
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
        corner_labels_all = [labels[i] for i in selected_idx] if selected_idx else labels

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
    new_wl = reso_range(mnest.param['min_wl'] - 0.05, mnest.param['max_wl'] + 0.05, 500, bins=True)
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
        cld_idx = np.where(np.diff(param['vmr_H2O']) != 0.)[0]
        if len(cld_idx) == 0:
            cld_idx = [len(param['P']) - 1]

        plt.hlines(
            param['P'][max(cld_idx)],
            ax.get_xlim()[0],
            ax.get_xlim()[1],
            linestyle='--',
            alpha=0.5,
            color='black',
            label='H$_2$O cloud',
        )
        plt.hlines(
            param['P'][min(cld_idx)],
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
        bottom = 5 * param['P'][max(cld_idx)]
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
            param['P'][max(cld_idx)],
            ax.get_xlim()[0],
            ax.get_xlim()[1],
            linestyle='--',
            alpha=0.5,
            color='black',
            label='H$_2$O cloud',
        )
        plt.hlines(
            param['P'][min(cld_idx)],
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
        bottom = 5 * param['P'][max(cld_idx)]
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


def plot_PT_profile(mnest, mc_samples, bestfit_cube, solutions=None):
    """Render the retrieved P–T profile and surface (P0, T0) credible contours."""
    del mc_samples  # PT plotting only uses the cached profile samples on disk.

    def _central_quantiles(sigmas):
        sigmas = np.asarray(sigmas, dtype=float)
        span = math.erf(sigmas / np.sqrt(2.0))
        return 0.5 - 0.5 * span, 0.5 + 0.5 * span

    def _sigma_percentile(sigma):
        return (0.5 + 0.5 * math.erf(float(sigma) / np.sqrt(2.0))) * 100.0

    def _credible_levels(density, masses):
        flat = density.ravel()
        order = np.argsort(flat)[::-1]
        cumulative = np.cumsum(flat[order])
        total = cumulative[-1]
        levels = []
        for mass in masses:
            target = mass * total
            idx = np.searchsorted(cumulative, target, side='left')
            if idx >= len(order):
                idx = len(order) - 1
            levels.append(flat[order[idx]])
        return np.asarray(levels)

    try:
        plt.style.use('seaborn-v0_8-white')
    except Exception:
        pass

    sample_path = mnest.param['out_dir'] + 'random_temp_samples.dat'
    if not os.path.isfile(sample_path):
        raise FileNotFoundError('Missing random_temp_samples.dat; run calc_spectra before plotting PT profiles.')

    samples = np.loadtxt(sample_path)
    pressure_grid = samples[2:, 0]
    profile_samples = samples[2:, 1:]
    surface_pressures = samples[0, 1:]
    surface_temperatures = samples[1, 1:]

    valid_rows = np.any(np.isfinite(profile_samples), axis=1)
    pressure_grid = pressure_grid[valid_rows]
    profile_samples = profile_samples[valid_rows]

    if pressure_grid.size == 0 or profile_samples.size == 0:
        raise ValueError('Temperature samples are empty; PT profile cannot be generated.')

    sigmas_shading = np.array([3.0, 2.0, 1.0], dtype=float)
    lower, upper = _central_quantiles(sigmas_shading)
    quantile_targets = list(lower) + [0.5] + list(upper[::-1])
    quantiles = np.nanquantile(profile_samples, quantile_targets, axis=1)
    q3_low, q2_low, q1_low, q50, q1_high, q2_high, q3_high = quantiles

    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=130)
    color = (64.0 / 255.0, 71.0 / 255.0, 132.0 / 255.0)

    ax.set_yscale('log')
    ax.invert_yaxis()
    ax.set_xlabel('Temperature [K]')
    ax.set_ylabel('Pressure [Pa]')

    ax.fill_betweenx(pressure_grid, q3_low, q3_high, color=color, alpha=0.25)
    ax.fill_betweenx(pressure_grid, q2_low, q2_high, color=color, alpha=0.40)
    ax.fill_betweenx(pressure_grid, q1_low, q1_high, color=color, alpha=0.65)

    mnest.cube_to_param(bestfit_cube)
    bestfit_profile = temp_profile(mnest.param)
    bestfit_pressures = np.asarray(mnest.param['P'])
    ax.plot(bestfit_profile, bestfit_pressures, linestyle='--', color='orange', lw=1.2)
    ax.scatter(bestfit_profile[-1], bestfit_pressures[-1], marker='s', fc='orange', ec='black', s=30, zorder=6)

    axis_top_candidates = [pressure_grid[0], bestfit_pressures[0]]
    axis_bottom_candidates = [
        pressure_grid[-1],
        bestfit_pressures[-1],
        np.nanpercentile(surface_pressures, _sigma_percentile(3.0)),
    ]
    axis_top_candidates = [val for val in axis_top_candidates if np.isfinite(val) and val > 0.0]
    axis_bottom_candidates = [val for val in axis_bottom_candidates if np.isfinite(val) and val > 0.0]

    inset_ax = ax.inset_axes([0.68, 0.64, 0.28, 0.28])
    mask_surface = np.isfinite(surface_temperatures) & np.isfinite(surface_pressures) & (surface_pressures > 0.0)
    temps = surface_temperatures[mask_surface]
    press = surface_pressures[mask_surface]
    sigmas_corner = np.array([0.5, 1.0, 2.0, 3.0], dtype=float)
    credible_masses = 1.0 - np.exp(-0.5 * sigmas_corner ** 2)

    if temps.size > 5 and press.size > 5:
        t_bins = np.linspace(np.nanmin(temps), np.nanmax(temps), 80)
        p_bins = np.logspace(np.log10(np.nanmin(press)), np.log10(np.nanmax(press)), 80)
        hist, t_edges, p_edges = np.histogram2d(temps, press, bins=[t_bins, p_bins])
        density = hist.T
        if density.sum() > 0.0:
            density /= density.sum()
            thresholds = _credible_levels(density, credible_masses)
            thresholds = np.clip(thresholds, np.finfo(float).eps, None)
            thresholds = np.unique(thresholds)
            t_centers = 0.5 * (t_edges[:-1] + t_edges[1:])
            p_centers = np.sqrt(p_edges[:-1] * p_edges[1:])
            TT, PP = np.meshgrid(t_centers, p_centers)
            fill_alphas = [0.60, 0.45, 0.30, 0.15]
            dens_max = density.max()
            upper_bound = dens_max * (1.0 + 1e-6)
            for level, alpha in zip(thresholds[::-1], fill_alphas):
                inset_ax.contourf(TT, PP, density, levels=[level, upper_bound], colors=[color], alpha=alpha)
            if thresholds.size > 0:
                inset_ax.contour(TT, PP, density, levels=np.sort(thresholds), colors=[color], linewidths=1.0)

    inset_ax.scatter(bestfit_profile[-1], bestfit_pressures[-1], marker='s', fc='orange', ec='black', s=30, zorder=7)
    inset_ax.set_xlabel(r'T$_0$ [K]', fontsize=9)
    inset_ax.set_ylabel(r'P$_0$ [Pa]', fontsize=9)
    inset_ax.set_yscale('log')
    inset_ax.invert_yaxis()
    inset_ax.tick_params(axis='both', which='both', labelsize=8)
    inset_ax.margins(x=0.05, y=0.05)

    legend_elements = [
        Patch(fc='none', ec='none', label='Retrieval:'),
        Line2D([], [], color='orange', linestyle='--', linewidth=1.2, label='Best fit model'),
        Patch(fc=(color[0], color[1], color[2], 0.65), ec='none', label=r'1$\sigma$'),
        Patch(fc=(color[0], color[1], color[2], 0.40), ec='none', label=r'2$\sigma$'),
        Patch(fc=(color[0], color[1], color[2], 0.25), ec='none', label=r'3$\sigma$'),
    ]

    if mnest.param['truths'] is not None:
        truths = np.genfromtxt(mnest.param['truths'])
        idx_PT_vars = 1 + len(mnest.param['fit_molecules']) + mnest.param['fit_ag']

        tmp_param = {
            'PT_profile_type': 'parametric',
            'Ts': mnest.param['Ts'],
            'Rs': mnest.param['Rs'],
            'major-a': mnest.param['major-a'],
            'P': 10.0 ** np.arange(0.0, truths[0] + 0.01, step=0.01),
            'kappa_th': 10 ** truths[idx_PT_vars],
            'gamma': 10 ** truths[idx_PT_vars + 1],
            'beta': truths[idx_PT_vars + 2],
            'Tint': truths[idx_PT_vars + 3] if mnest.param['fit_Tint'] else mnest.param['Tint'],
        }

        if mnest.param['fit_p_size']:
            locate_mp_rp = 4
        else:
            locate_mp_rp = 3

        if mnest.param['fit_g']:
            tmp_param['gp'] = truths[-(locate_mp_rp - 1)]
        else:
            if mnest.param['fit_Mp']:
                Mp = truths[-locate_mp_rp] * const.M_earth.value
            else:
                Mp = mnest.param['Mp'] * const.M_jup.value
            if mnest.param['fit_Rp']:
                Rp = truths[-(locate_mp_rp - 1)] * const.R_earth.value
            else:
                Rp = mnest.param['Rp'] * const.M_jup.value
            tmp_param['gp'] = (const.G.value * Mp) / (Rp ** 2.0)

        truth_profile = temp_profile(tmp_param)
        truth_pressure = tmp_param['P']
        truth_surface_pressure = 10 ** truths[0]

        ax.plot(truth_profile, truth_pressure, linestyle='-', color='black', lw=1.0)
        ax.scatter(truth_profile[-1], truth_surface_pressure, marker='s', fc='white', ec='black', s=30, zorder=7)
        inset_ax.scatter(truth_profile[-1], truth_surface_pressure, marker='s', fc='white', ec='black', s=30, zorder=8)

        axis_top_candidates.append(truth_pressure[0])
        axis_bottom_candidates.append(truth_surface_pressure)

        legend_elements.extend([
            Patch(fc='none', ec='none', label='Truth:'),
            Line2D([], [], linestyle='-', color='black', lw=1.0, label='P-T profile'),
            Line2D([], [], marker='s', linestyle='', color='black', markerfacecolor='white', markeredgecolor='black', ms=5, label='Surface'),
        ])

    axis_top_candidates = [val for val in axis_top_candidates if np.isfinite(val) and val > 0.0]
    axis_bottom_candidates = [val for val in axis_bottom_candidates if np.isfinite(val) and val > 0.0]
    axis_top = min(axis_top_candidates)
    axis_bottom = max(axis_bottom_candidates) * 1.2
    ax.set_ylim(axis_bottom, axis_top)
    ax.autoscale(axis='x', tight=True)
    ax.margins(x=0.05)

    legend = ax.legend(handles=legend_elements, loc='lower left', frameon=False, fontsize=9)
    if legend is not None:
        legend._legend_box.align = 'left'

    fig.tight_layout()

    if solutions is None:
        fig.savefig(mnest.param['out_dir'] + 'PT_profile.pdf')
    else:
        fig.savefig(mnest.param['out_dir'] + f'PT_profile (solution {solutions}).pdf')

    plt.close(fig)
