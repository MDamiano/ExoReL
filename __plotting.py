import sys
import numpy as np
import matplotlib.pyplot as plt

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

