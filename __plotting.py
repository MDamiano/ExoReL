import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

from .__utils import find_nearest


def plot_chemistry(param, solutions=None):
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
