from .__basics import *
import re
import shutil


def default_parameters():
    param = {}

    param['wkg_dir'] = os.getcwd()

    #### [STAR] ####
    param['Rs'] = None  # Star radius
    param['Ts'] = None  # Star temperature
    param['meta'] = None  # star metallicity [M / H]
    param['Loggs'] = None  # Star log surface gravity
    param['stellar_dir'] = None  # optional directory containing high-resolution SVO stellar spectra
    param['distance'] = None  # star distance from the Sun [pc]

    #### [PLANET] ####
    param['name_p'] = None  # Planet name
    param['major-a'] = None  # Planet semi-major axis
    param['eccentricity'] = None  # Planet eccentricity
    param['inclination'] = 90.0  # Planet inclination. 90.0 deg -> edge on. 0.0 deg -> face on [deg]
    param['Rp'] = None  # Planet radius [Jupiter radii]
    param['Rp_err'] = None  # Planet radius error
    param['Mp'] = None  # Planet mass [Jupiter mass]
    param['Mp_err'] = None  # Planet mass error
    param['gp'] = None  # Planet surface gravity [m/s^2]
    param['Tp'] = None  # Planet surface temperature [K]
    param['Tirr'] = 394.109  # Irradiation Temperature at 1 AU related to the Sun case [K]
    param['Tint'] = 110.0  # Intrinsic (internal) Temperature [K]
    param['phi'] = None  # Phase angle [deg]
    param['phi_err'] = None  # Phase angle error [deg]
    param['P0'] = None  # Surface pressure [Pa]
    param['Ag'] = None  # Surface albedo

    #### [ATMOSPHERIC_PAR] ####
    param['fhaze'] = 1e-36  # flux haze -- NOT IN USE
    param['cld_frac'] = 1.0  # cloud fraction
    param['adjust_VMR_gases'] = True  # All the gases are adjusted to compensate water condensation depletion
    param['use_adaptive_grid'] = True  # Split the atmosphere altitude in the same number of layers below, within, and above the clouds
    param['n_layer'] = 100  # Number of layers of the atmosphere
    param['KE'] = 1.0  # Eddy diffusion coefficient in m2/s
    param['opar'] = 3.0  # correct the Rossland mean opacity at low pressure

    #### [MODEL_PAR] ####
    param['physics_model'] = 'radiative_transfer'  # choose between 'radiative_transfer', 'dataset', or 'AI_model'
    param['physics_model_code_language'] = 'Python' # choose between 'C' and 'Python'
    param['opac_data'] = '2k'  # spectral resolution of the opacities
    param['opac_dir'] = None  # directory containing the opacities (this folder will have subdirectories for different spectral resolutions)
    param['use_float32'] = True  # whether or not to use float32 or float64 for the opacities precision
    param['gaseous_planet'] = False  # whether the planet is gaseous or rocky. If False, the surface pressure and albedo will be included in the model and can be fit during retrieval
    param['P_standard'] = 10. ** np.arange(0.0, 12.01, step=0.01)  # standard pressure grid in Pa
    param['fit_p0'] = False  # whether to fit the surface parameter during retrieval
    param['fit_ag'] = False  # whether to fit the surface albedo during retrieval
    param['surface_albedo_parameters'] = 1  # how many different surface albedo to fit if 'param['fit_ag']' is True (choose between 1, 3, 5)
    param['gas_par_space'] = None  # which space definition to use to fit the gases (choose between 'partial_pressure', 'centered_log_ratio', 'clr', 'volume_mixing_ratio', or 'vmr')
    param['mod_prior'] = True  # If 'clr' or 'centered_log_ratio' is chosen as space, then use the modified prior introduced in Damiano & Hu 2021
    param['supported_molecules'] = ['H2O', 'NH3', 'CH4', 'H2S', 'SO2', 'CO2', 'CO', 'O2', 'O3', 'N2O', 'N2', 'He', 'H2']
    for i in param['supported_molecules']:
        param['fit_' + i] = False
    param['H2_He_ratio'] = 0.85  # Hydrogen - Helium ratio of the filling portion of the atmosphere
    param['O3_earth'] = False  # whether the O3 VMR is limited between two atmospheric pressure; 0 outside the band.
    param['gas_fill'] = None  # which gas to consider as filler
    param['fit_phi'] = False  # whether to fit the orbital phase angle during retrieval
    param['fit_g'] = False  # whether to fit the planetary surface gravity during retrieval
    param['fit_Mp'] = False  # whether to fit the planetary mass during retrieval
    param['fit_Rp'] = False  # whether to fit the planetary radius during retrieval
    param['fit_T'] = False  # whether to fit the planetary temperature during retrieval
    param['PT_profile_type'] = 'isothermal'  # type of PT profile to use. Possibilities: isothermal, parametric
    param['Rp_prior_type'] = None  # type of prior function for the planetary radius. Possibilities: independent, M_R_prior
    param['Mp_prior_type'] = None  # type of prior function for the planetary mass. Possibilities: independent, M_R_prior, gaussian

    param['fit_wtr_cld'] = False  # whether to include and fit water cloud position during retrieval
    param['wtr_cld_type'] = 'liquid'  # type of water cloud to consider. Choose between 'liquid', 'ice', and 'mixed'
    param['fit_amm_cld'] = False  # whether to include and fit ammonia cloud position during retrieval
    param['fit_cld_frac'] = False  # whether to fit the cloud fraction during retrieval. The cloud fraction is applied to all clouds present in the model

    param['fit_p_size'] = False  # whether to fit particle size during retrieval
    param['p_size_type'] = 'constant'  # type of particle size fitting. (choose between constant or factor)
    param['albedo_calc'] = False  # whether the model return the albedo spectrum as output
    param['fp_over_fs'] = False  # whether the model return the contrast ratio spectrum as output
    param['flat_albedo'] = False  # whether to use a flat albedo model for the planet
    param['flat_albedo_value'] = None  # which value to use for the flat albedo calculation [0.0, 1.0]
    param['hazes'] = False  # whether to include and fit hazes during the retrieval

    #### [MISC_PAR] ####
    param['output_directory'] = None  # name of the output directory where all the results will be stored. If None, the output will be stored in the working directory
    param['file_output_name'] = None  # name of the output file for the spectrum and posterior distribution. If None, the output file will be named "spectrum.dat"
    param['obs_numb'] = None  # Number of observations to be taken into account during retrieval
    param['optimizer'] = None  # Which optimizer to use during retrieval. 'multinest' is the only possibility currently
    param['gen_dataset_mode'] = False
    param['random_seed'] = None  # Random seed for GEN_DATASET design matrix generation

    #### [MULTINEST_PAR] ####
    param['multimodal'] = True
    param['max_modes'] = 100
    param['ev_tolerance'] = 0.5
    param['nlive_p'] = 2000
    param['multinest_resume'] = True
    param['multinest_verbose'] = False

    #### [Plotting_Options_PAR] ####
    param['wl_native'] = False  # use the opacity wl grid for the output
    param['mol_custom_wl'] = False  # use a custom wl grid for the output
    param['filter_multi_solutions'] = False  # whether to filter low Bayesian evidence solutions
    param['plot_contribution'] = False  # whether to plot the spectral contribution of the individual gases
    param['plot_posterior'] = False  # whether to plot the marginalized posterior distribution functions
    param['corner_selected_params'] = None  # list of parameter indices to plot in the corner plot
    param['truths'] = None  # whether to also plot the truths value in the posterior plot
    param['calc_likelihood_data'] = False
    param['n_likelihood_data'] = 10240
    param['plot_elpd_stats'] = False  # whether to calculate and plot the expected log pointwise predictive density statistics
    param['elpd_reference'] = None # path to the reference dataset for elpd calculation

    #### [Create_spectrum_PAR] ####
    param['add_noise'] = False
    param['gaussian_noise'] = False
    param['noise_model'] = 0
    param['save_snr_array'] = False
    param['snr'] = 20
    param['return_bins'] = False

    mm = {'H': 1.00784, 'He': 4.002602, 'C': 12.0107, 'N': 14.0067, 'O': 15.9994, 'S': 32.065}
    mm['H2'] = mm['H'] * 2.
    mm['N2'] = mm['N'] * 2.
    mm['O2'] = mm['O'] * 2.
    mm['O3'] = mm['O'] * 3.
    mm['CO'] = mm['C'] + mm['O']
    mm['H2O'] = (mm['H'] * 2.) + mm['O']
    mm['H2S'] = (mm['H'] * 2.) + mm['S']
    mm['N2O'] = (mm['N'] * 2.) + mm['O']
    mm['CO2'] = mm['C'] + (mm['O'] * 2.)
    mm['SO2'] = mm['S'] + (mm['O'] * 2.)
    mm['NH3'] = mm['N'] + (mm['H'] * 3.)
    mm['CH4'] = mm['C'] + (mm['H'] * 4.)

    param['mm'] = mm

    param['formatted_labels'] = {}
    for mol in param['supported_molecules']:
        if mol == 'H2O':
            param['formatted_labels'][mol] = "Log(H$_2$O)"
        if mol == 'NH3':
            param['formatted_labels'][mol] = "Log(NH$_3$)"
        if mol == 'CH4':
            param['formatted_labels'][mol] = "Log(CH$_4$)"
        if mol == 'H2S':
            param['formatted_labels'][mol] = "Log(H$_2$S)"
        if mol == 'SO2':
            param['formatted_labels'][mol] = "Log(SO$_2$)"
        if mol == 'CO2':
            param['formatted_labels'][mol] = "Log(CO$_2$)"
        if mol == 'CO':
            param['formatted_labels'][mol] = "Log(CO)"
        if mol == 'O2':
            param['formatted_labels'][mol] = "Log(O$_2$)"
        if mol == 'O3':
            param['formatted_labels'][mol] = "Log(O$_3$)"
        if mol == 'N2O':
            param['formatted_labels'][mol] = "Log(N$_2$O)"
        if mol == 'N2':
            param['formatted_labels'][mol] = "Log(N$_2$)"
        if mol == 'H2':
            param['formatted_labels'][mol] = "Log(H$_2$)"

    return param


def read_parfile(param, parfile=None):
    cwd = os.getcwd()
    if parfile is None:
        raise ValueError('A parameter file path must be provided.')

    if os.path.isabs(parfile):
        parfile_path = parfile
    else:
        parfile_path = os.path.join(cwd, parfile)

    with open(parfile_path, 'r') as f:
        paramdata = json.load(f)
    for key, value in paramdata.items():
        param[key] = value
    del paramdata

    param['parfile_path'] = os.path.abspath(parfile_path)
    param['wkg_dir'] = cwd + '/'

    return param


def copy_parfile_to_output(param):
    parfile_path = param.get('parfile_path')
    out_dir = param.get('out_dir')
    if not parfile_path or not out_dir:
        return None

    source = os.path.abspath(parfile_path)
    destination = os.path.join(out_dir, os.path.basename(source))
    if os.path.normcase(source) == os.path.normcase(os.path.abspath(destination)):
        return destination

    shutil.copy2(source, destination)
    return destination


def setup_param_dict(param):
    if param['obs_numb'] is not None:
        param['obs_numb'] = int(param['obs_numb'])

    param['contribution'] = False
    param['mol_contr'] = None

    if param['albedo_calc']:
        param['fp_over_fs'] = False

    if param['fit_wtr_cld'] and param['fit_amm_cld']:
        param['double_cloud'] = True
    else:
        param['double_cloud'] = False

    if param['gaseous_planet']:
        param['rocky'] = False
        param['fit_p0'] = False
        param['P0'] = 10 ** 11.5
        param['fit_ag'] = False
        param['Ag'] = 0.0
        param['gas_par_space'] = 'vmr'
        param['gas_fill'] = 'H2'
    else:
        param['rocky'] = True

    if param['fit_Rp'] and param['Rp'] is not None and param['Rp_err'] is not None:
        param['Rp_orig'] = param['Rp'] + 0.0

    if param['Mp'] is not None:
        param['Mp_orig'] = param['Mp'] + 0.0

    if param['fit_phi'] and param['phi'] is not None and param['phi_err'] is not None:
        param['phi_orig'] = param['phi'] + 0.0

    if param['gas_fill'] == 'N2':
        param['fit_N2'] = False

    if 'vmr_range' in param.keys():
        param['gas_par_space'] = 'vmr'
    elif 'clr_range' in param.keys():
        param['gas_par_space'] = 'clr'
    elif 'pp_range' in param.keys():
        param['gas_par_space'] = 'partial_pressure'

    if param['rocky'] and not param['fit_p0'] and param['P0'] is None and param['gas_par_space'] != 'partial_pressure':
        raise ValueError("Surface pressure (P0) needs to be specified since it is not a free parameter.")

    if param['gas_par_space'] == 'partial_pressure' and param['fit_p0']:
        param['fit_p0'] = False
        print('The parameter "fit_p0" has been set to False since the atmospheric chemistry will be fit in the "partial pressure" parameter space.')

    if param['rocky'] and not param['fit_ag']:
        if param['surface_albedo_parameters'] == int(1) and param['Ag'] is None:
            raise ValueError("Surface albedo (Ag) needs to be specified since it is not a free parameter.")
        elif param['surface_albedo_parameters'] == int(3) and (param['Ag1'] is None or param['Ag2'] is None or param['Ag_x1'] is None):
            raise ValueError("Surface albedo parameters (Ag1, Ag2, Ag_x1) need to be specified since they are not free parameters and the number of parameters is set to 3.")
        elif param['surface_albedo_parameters'] == int(5) and (param['Ag1'] is None or param['Ag2'] is None or param['Ag3'] is None or param['Ag_x1'] is None or param['Ag_x2'] is None):
            raise ValueError("Surface albedo parameters (Ag1, Ag2, Ag3, Ag_x1, Ag_x2) need to be specified since they are not free parameters and the number of parameters is set to 5.")
        else:
            pass

    if param['rocky'] and param['fit_ag'] and param['surface_albedo_parameters'] is None:
        param['surface_albedo_parameters'] = int(1)
        print('Surface albedo parameters number not defined. The parameter "surface_albedo_parameters" has been set to 1.')

    if not param['fit_g'] and not param['fit_Mp'] and not param['fit_Rp']:
        if (param['Rp'] is not None) or ('Rp_range' in param.keys()):
            param['Rp_provided'] = True
        else:
            param['Rp_provided'] = False

        if (param['Mp'] is not None) or ('Mp_range' in param.keys()):
            param['Mp_provided'] = True
        else:
            param['Mp_provided'] = False

        if param['gp'] is not None:
            param['gp_provided'] = True
        else:
            param['gp_provided'] = False

        if param['Rp'] is None and param['gp'] is None:
            if not param['Rp_provided']:
                raise ValueError("If radius, mass, and gravity of the planet are not free parameters, please provide at least a combination of two in the parameter file.")

    if param['PT_profile_type'] == 'parametric':
        param['wtr_cld_type'] = 'mixed'

    if param['cld_frac'] > 1.0 or param['cld_frac'] < 0.0:
        raise ValueError("The cloud fraction should be defined between [0.0, 1.0]. Please check the 'cld_frac' value in the parameter file.")

    if param['optimizer'] == 'multinest':
        param['nlive_p'] = int(param['nlive_p'])
        param['max_modes'] = int(param['max_modes'])
    elif param['optimizer'] == 'sobol' or param['optimizer'] == 'random':
        param['n_spectra'] = int(param['n_spectra'])
    else:
        pass

    if param['optimizer'] is not None:
        param['fit_molecules'] = []
        for mol in param['supported_molecules']:
            param['vmr_' + mol] = 0.0
            if param['fit_' + mol]:
                param['fit_molecules'].append(mol)

    param['n_layer'] = int(param['n_layer'])
    param['out_dir'] = resolve_output_dir(param)

    return param


def resolve_output_dir(param):
    output_directory = param.get('output_directory')
    if output_directory is not None:
        out_dir = os.path.join(param['wkg_dir'], output_directory)
    else:
        out_dir = param['wkg_dir']

    return os.path.normpath(out_dir) + os.sep


def par_and_calc(param):
    # star
    param['Ls'] = (param['Rs'] ** 2.) * ((param['Ts'] / 5760.) ** 4.)

    # planet
    if not param['fit_T']:
        if param['PT_profile_type'] == 'isothermal':
            try:
                param['Tp'] += 0.0
            except (KeyError, TypeError):
                t1 = ((param['Rs'] * const.R_sun.value) / (2. * param['major-a'] * const.au.value)) ** 0.5
                param['Tp'] = t1 * ((1 - 0.3) ** 0.25) * param['Ts']
        elif param['PT_profile_type'] == 'parametric':
            param['kappa_th'] = 10. ** param['kappa_th']
            param['gamma'] = 10. ** param['gamma']
        else:
            PT_prof = np.genfromtxt(param['PT_profile_type'])
            param['Pp'] = PT_prof[:,0] + 0.0
            param['Tp'] = PT_prof[:,1] + 0.0

    # Insolation variation
    if param['eccentricity'] != 0.0:
        F_min = param['Ls'] / (param['major-a'] * (1. + param['eccentricity'])) ** 2.
        F_max = param['Ls'] / (param['major-a'] * (1. - param['eccentricity'])) ** 2.
        F_ave = (F_min + F_max) / 2.0
    else:
        F_ave = param['Ls'] / (param['major-a'] ** 2.)

    # Equivalent semi - major axis
    a_ave = 1. / (F_ave ** 0.5)
    param['equivalent_a'] = a_ave
    param['F_ave'] = F_ave
    param['Tirr'] /= (param['equivalent_a'] ** 0.5)

    if param['obs_numb'] is None:
        if not param['fit_phi']:
            param['phi'] = math.pi * param['phi'] / 180.0
    else:
        for obs in range(0, param['obs_numb']):
            if not param['fit_phi']:
                param['phi' + str(obs)] = math.pi * param['phi' + str(obs)] / 180.0

    return param


def calc_mean_mol_mass(param):
    param['mean_mol_weight'] = np.zeros(len(param['P']))
    for i in range(0, len(param['P'])):
        for mol in param['fit_molecules']:
            param['mean_mol_weight'][i] += param['vmr_' + mol][i] * param['mm'][mol]
        if param['gas_fill'] is not None:
            if param['rocky']:
                param['mean_mol_weight'][i] += param['vmr_' + param['gas_fill']][i] * param['mm'][param['gas_fill']]
            else:
                param['mean_mol_weight'][i] += (param['vmr_' + param['gas_fill']][i] * param['mm'][param['gas_fill']]) + (param['vmr_He'][i] * param['mm']['He'])

    if not param['ret_mode'] and param['verbose']:
        print('mu \t\t = \t' + str(param['mean_mol_weight'][-1]))

    return param


def load_input_spectrum(param):
    if param['ret_mode']:
        try:
            if param['obs_numb'] is None:
                spectrum = np.loadtxt(param['wkg_dir'] + param['spectrum'])
                if spectrum.ndim == 1:
                    spectrum = spectrum.reshape(1, -1)
                param['spectrum'] = {}
                if spectrum.shape[1] == 3:
                    param['spectrum']['wl'] = spectrum[:, 0]            # wavelength in micron
                    param['spectrum']['Fplanet'] = spectrum[:, 1]       # (W/m2) or contrast ratio
                    param['spectrum']['error_p'] = spectrum[:, 2]       # (W/m2) or contrast ratio
                    param['spectrum']['bins'] = False
                    param['min_wl'] = min(spectrum[:, 0])
                    param['max_wl'] = max(spectrum[:, 0])
                else:
                    param['spectrum']['wl_low'] = spectrum[:, 0]        # wavelength bin_low in micron
                    param['spectrum']['wl_high'] = spectrum[:, 1]       # wavelength bin_high in micron
                    param['spectrum']['wl'] = spectrum[:, 2]            # wavelength in micron
                    param['spectrum']['Fplanet'] = spectrum[:, 3]       # (W/m2) or contrast ratio
                    param['spectrum']['error_p'] = spectrum[:, 4]       # (W/m2) or contrast ratio
                    param['spectrum']['bins'] = True
                    param['min_wl'] = min(param['spectrum']['wl_low'])
                    param['max_wl'] = max(param['spectrum']['wl_high'])
            else:
                param['spectrum'] = {}
                min_wl, max_wl = [], []
                for obs in range(0, int(param['obs_numb'])):
                    spectrum = np.loadtxt(param['wkg_dir'] + param['spectrum' + str(obs)])
                    if spectrum.ndim == 1:
                        spectrum = spectrum.reshape(1, -1)
                    param['spectrum'][str(obs)] = {}
                    if spectrum.shape[1] == 3:
                        param['spectrum'][str(obs)]['wl'] = spectrum[:, 0]
                        param['spectrum'][str(obs)]['Fplanet'] = spectrum[:, 1]
                        param['spectrum'][str(obs)]['error_p'] = spectrum[:, 2]
                        param['spectrum']['bins'] = False
                        min_wl.append(float(min(spectrum[:, 0])))
                        max_wl.append(float(max(spectrum[:, 0])))
                    else:
                        param['spectrum'][str(obs)]['wl_low'] = spectrum[:, 0]          # wavelength bin_low in micron
                        param['spectrum'][str(obs)]['wl_high'] = spectrum[:, 1]         # wavelength bin_high in micron
                        param['spectrum'][str(obs)]['wl'] = spectrum[:, 2]              # wavelength in micron
                        param['spectrum'][str(obs)]['Fplanet'] = spectrum[:, 3]         # (W/m2) or contrast ratio
                        param['spectrum'][str(obs)]['error_p'] = spectrum[:, 4]         # (W/m2) or contrast ratio
                        param['spectrum']['bins'] = True
                        min_wl.append(float(min(spectrum[:, 0])))
                        max_wl.append(float(max(spectrum[:, 1])))
                param['min_wl'] = min(np.array(min_wl))
                param['max_wl'] = max(np.array(max_wl))
        except KeyError:
            print('An input spectrum is required, in the parameter file, use the "spectrum" keyword followed by the path of the file')
            sys.exit()
    else:
        try:
            spectrum = np.loadtxt(param['pkg_dir'] + 'forward_mod/Data/wl_bins/' + param['wave_file'] + '.dat')
        except KeyError:
            if param['rocky']:
                # standard wavelength bin at R = 500 in 0.15 - 2.0 micron
                spectrum = np.loadtxt(param['pkg_dir'] + 'forward_mod/Data/wl_bins/bins_02_20_R500.dat')
            else:
                # standard wavelength bin at R = 500 in the optical wavelength 0.4 - 1.0 micron
                spectrum = np.loadtxt(param['pkg_dir'] + 'forward_mod/Data/wl_bins/bins_04_10_R500.dat')
        except FileNotFoundError:
            print('File "' + param['pkg_dir'] + 'forward_mod/Data/wl_bins/' + param['wave_file'] + '.dat" not found. Using the native wavelength bins of opacities.')
            param['wl_native'] = True
            spectrum = np.loadtxt(param['pkg_dir'] + 'forward_mod/Data/wl_bins/bins_02_20_R500.dat')

        param['spectrum'] = {}
        try:
            param['spectrum']['wl_low'] = spectrum[:, 0] + 0.0  # wavelength bin_low in micron
            param['spectrum']['wl_high'] = spectrum[:, 1] + 0.0  # wavelength bin_high in micron
            param['spectrum']['wl'] = np.mean(np.array([param['spectrum']['wl_low'], param['spectrum']['wl_high']]).T, axis=1)  # wavelength in micron
            param['spectrum']['bins'] = True
            param['min_wl'] = min(param['spectrum']['wl_low'])
            param['max_wl'] = max(param['spectrum']['wl_high'])
        except IndexError:
            param['spectrum']['wl'] = spectrum
            param['spectrum']['bins'] = False
            param['min_wl'] = min(param['spectrum']['wl'])
            param['max_wl'] = max(param['spectrum']['wl'])

    if param['physics_model'] == 'radiative_transfer' and param['physics_model_code_language'] == 'C':
        param['wl_C_grid'] = (10. ** np.linspace(np.log10(1e-7), np.log10(2e-4), 16000)) * 1e6
        param['start_c_wl_grid'] = find_nearest(param['wl_C_grid'], param['min_wl']) - 35
        param['stop_c_wl_grid'] = find_nearest(param['wl_C_grid'], param['max_wl']) + 35
    return param


def find_nearest(array, value):
    idx = np.nanargmin(np.absolute(array - value))
    return idx


def alphabet():
    alfalecter = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x', 'y', 'w', 'z']
    return alfalecter[random.randint(0, len(alfalecter)-1)]


def particlesizef(g, T, P, M, MM, KE, deltaP):
    # Calculate particle size in exoplanet atmospheres

    # input
    # g in SI
    # T in K
    # P in Pa
    # M: mean molecular mass of the atmosphere; in g / mol
    # MM: molecular mass of the condensable species; in g / mol
    # KE: Eddy diffusion coefficient; in m2 s - 1
    # deltaP: difference between partial pressure and saturation vapor
    # pressure, in Pa

    # assume
    # density of condensed material of water 1000 kg / m3
    # accomodation factor of unity
    # sig = 2

    # output particle size in micron, and volumn in cm ^ 3

    # Derived parameters
    H = (const.k_B.value * T) / M / const.u.value / g
    u = KE / H
    mu = ((8.76E-6 * (293.85 + 72)) / (293.85 + 72)) * ((T / 293.85) ** 1.5)  # SI
    lamb = (2. * mu) / P / ((8 * M * 1.0E-3 / math.pi / 8.314472 / T) ** 0.5)  # m
    # KK = 4 * KB * T / 3. / mu
    deltan = deltaP / const.k_B.value / T

    # droplet
    rho = 1.0E+3  # kg m-3
    acc = 1.0

    # mass diffusion coefficient
    D = 0.12E-4

    # Particle Size and Number
    Cc, fa = 1, 1
    Cc1, fa1 = 2, 2
    sig = 2

    check = 0
    while (abs(Cc1 - Cc) + abs(fa1 - fa)) > 0.001:
        Cc = Cc1 + 0.0
        fa = fa1 + 0.0
        cc = -((48. * math.pi ** 2.) ** (1. / 3.)) * D * MM * const.u.value * fa * deltan / rho * np.exp(- np.log(sig) ** 2.)  # effective condensation coefficient D
        aa = rho * g / mu / ((162. * math.pi ** 2.) ** (1. / 3.)) / H * Cc * np.exp(- np.log(sig) ** 2.)
        bb = -u / H

        V = ((-bb + np.sqrt((bb ** 2.) - (4. * aa * cc))) / 2. / aa) ** (3. / 2.)
        d1 = ((6. * V / math.pi) ** (1. / 3.)) * np.exp(- np.log(sig) ** 2.)

        kn = lamb / d1
        Cc1 = 1. + kn * (1.257 + 0.4 * np.exp(- 1.1 / kn))
        fa1 = (1. + kn) / (1. + 2. * kn * (1. + kn) / acc)

        Vs = V + 0.0
        check += 1
        if check > 1e4:
            break

    r0 = (3. * Vs / 4. / math.pi) ** (1. / 3.) * np.exp(- 1.5 * np.log(sig) ** 2.) * 1.0E+6
    r1 = (3. * Vs / 4. / math.pi) ** (1. / 3.) * np.exp(- 1.0 * np.log(sig) ** 2.) * 1.0E+6
    r2 = (3. * Vs / 4. / math.pi) ** (1. / 3.) * np.exp(- 0.5 * np.log(sig) ** 2.) * 1.0E+6
    VP = Vs * 1.0E+6

    return r0, r1, r2, VP


def cloud_pos(param, condensed_gas='H2O'):
    def waterpressure(t):
        # Saturation Vapor Pressure of Water
        # t in K
        # p in Pascal

        try:
            p = np.empty((len(t)))
        except TypeError:
            t = np.array([t])
            p = np.empty(len(t))

        for i in range(0, len(t)):
            if t[i] < 273.16:
                # Formulation from Murphy & Koop(2005)
                p[i] = np.exp(9.550426 - (5723.265 / t[i]) + (3.53068 * np.log(t[i])) - (0.00728332 * t[i]))
            elif t[i] < 373.15:
                # Formulation from Seinfeld & Pandis(2006)
                a = 1 - (373.15 / t[i])
                p[i] = 101325. * np.exp((13.3185 * a) - (1.97 * (a ** 2.)) - (0.6445 * (a ** 3.)) - (0.1229 * (a ** 4.)))
            elif t[i] < 647.09:
                p[i] = (10. ** (8.14019 - (1810.94 / (244.485 + t[i] - 273.15)))) * 133.322387415
            else:
                p[i] = np.nan
        return p

    def ammoniapressure(t):
        # Saturation Vapor Pressure of Ammonia
        # tl in K
        # psat in Pascal
        # Lodders, K., Fegley Jr., B., 1998. The Planetary Scientist?s Companion. Oxford Univ. Press, 371 pp.
        
        tl_array = np.asarray(t, dtype=float)
        tl_flat = tl_array.reshape(-1)
        psat_flat = np.full_like(tl_flat, np.nan, dtype=float)

        solid = tl_flat < 195.4
        if solid.any():
            psat_flat[solid] = 10.0 ** (6.9 - 1588.0 / tl_flat[solid]) * 1.0e5

        liquid = (tl_flat >= 195.4) & (tl_flat < 300.0)
        if liquid.any():
            psat_flat[liquid] = 10.0 ** (5.201 - 1248.0 / tl_flat[liquid]) * 1.0e5

        # Ackerman & Marley
        # psat(i)=exp(10.53-2161/tl(i)-86596/tl(i)/tl(i))*1E+5;

        psat = psat_flat.reshape(tl_array.shape)
        return psat.item() if psat.shape == () else psat

    if condensed_gas == 'H2O':
        short_name = 'wtr'
        initial_letter = 'w'
    elif condensed_gas == 'NH3':
        short_name = 'amm'
        initial_letter = 'a'

    if param['fit_' + short_name + '_cld']:
        if param['PT_profile_type'] == 'parametric':
            P = param['P']

            if short_name == 'wtr':
                psat = waterpressure(param['T'])
            elif short_name == 'amm':
                psat = ammoniapressure(param['T'])
            base_profile = np.asarray(param['vmr_' + condensed_gas], dtype=float)
            if base_profile.ndim == 0:
                base_profile = np.full(len(P), float(base_profile))
            elif base_profile.size == len(param['P_standard']):
                base_profile = base_profile[:len(P)].astype(float, copy=True)
            elif base_profile.size != len(P):
                raise ValueError(f"vmr_{condensed_gas} size does not match the active pressure grid.")
            mix = np.empty((len(P)))
            # The condensable abundance may already be a pressure profile if
            # another cloud species was processed earlier.
            mix[-1] = np.nanmin([psat[-1] / P[-1], base_profile[-1]])
            for i in range(len(P)-2, -1, -1):
                mix[i] = np.nanmin([psat[i] / P[i], base_profile[i], mix[i+1]])
        else:
            base_profile = np.asarray(param['vmr_' + condensed_gas], dtype=float)
            if base_profile.ndim == 0:
                base_profile = np.full(len(param['P']), float(base_profile))
            elif base_profile.size == len(param['P_standard']):
                base_profile = base_profile[:len(param['P'])].astype(float, copy=True)
            elif base_profile.size != len(param['P']):
                raise ValueError(f"vmr_{condensed_gas} size does not match the active pressure grid.")

            # if param['Pw_top'] > param['P'][-1]:
            if param['P' + initial_letter + '_top'] > param['P'][-1] or (param['P' + initial_letter + '_top'] + param['cld' + initial_letter + '_depth']) > param['P'][-1]:
                no_cloud = True
            else:
                no_cloud = False

            if not no_cloud:
                pos_cld = int(find_nearest(param['P'], param['P' + initial_letter + '_top']))

                if (param['cld' + initial_letter + '_depth'] + param['P'][pos_cld]) > param['P'][-1]:
                    param['cld' + initial_letter + '_depth'] = param['P'][-1] - param['P'][pos_cld]

                pbot = int(find_nearest(param['P'], (param['cld' + initial_letter + '_depth'] + param['P'][pos_cld])))

                depth_layers = pbot - pos_cld
                if depth_layers == 0:
                    return base_profile
                else:
                    pass

                mix_scale = np.ones(len(param['P'])) * param['CR_' + condensed_gas]
                d = -np.log10(param['CR_' + condensed_gas]) / depth_layers
                for i in range(0, len(mix_scale)):
                    if i <= pos_cld:
                        pass
                    elif pos_cld < i <= pos_cld + depth_layers:
                        mix_scale[i] = 10. ** (np.log10(mix_scale[i - 1]) + d)
                    elif i > pos_cld + depth_layers:
                        mix_scale[i] = mix_scale[i - 1]
                mix = base_profile * mix_scale
            else:
                mix = base_profile
    else:
        mix = np.ones((len(param['P']))) * param['vmr_' + condensed_gas]

    return mix


def adjust_VMR(param, all_gases=True, condensed_gas='H2O'):
    def _profile_for(mol, size):
        values = np.asarray(param['vmr_' + mol], dtype=float)
        if values.ndim == 0:
            return np.full(size, float(values))
        return values.astype(float, copy=True)

    if condensed_gas is None:
        if param['gas_fill'] is not None:
            gas_to_consider = param['fit_molecules'] + [param['gas_fill']]
        else:
            gas_to_consider = param['fit_molecules']

        for mol in gas_to_consider:
            param['vmr_' + mol] = np.ones((len(param['P']))) * param['vmr_' + mol]

    else:
        if all_gases:
            if param['gas_fill'] is None:
                mol_to_determine = [mol for mol in param['fit_molecules'] if mol != condensed_gas]
            else:
                mol_to_determine = [mol for mol in param['fit_molecules'] if mol != condensed_gas]
                mol_to_determine.append(param['gas_fill'])

            n_gases = len(mol_to_determine)
            if n_gases == 0:
                return param

            condensed_profile = np.atleast_1d(np.asarray(param['vmr_' + condensed_gas], dtype=float))
            source_profiles = {mol: _profile_for(mol, condensed_profile.size) for mol in mol_to_determine}
            for mol, values in source_profiles.items():
                if values.size != condensed_profile.size:
                    raise ValueError(f"vmr_{mol} size does not match vmr_{condensed_gas}.")
                param['vmr_' + mol] = np.zeros(condensed_profile.size)

            for i in range(condensed_profile.size):
                current = np.array([source_profiles[mol][i] for mol in mol_to_determine], dtype=float)
                v_m_r = np.zeros(n_gases)
                remaining_vmr = 1.0 - condensed_profile[i]

                positive = np.isfinite(current) & (current > 0.0)
                if np.any(positive):
                    v_m_r[positive] = remaining_vmr * current[positive] / np.sum(current[positive])
                else:
                    if param['gas_fill'] in mol_to_determine:
                        fill_idx = mol_to_determine.index(param['gas_fill'])
                    else:
                        fill_idx = 0
                    v_m_r[fill_idx] = remaining_vmr

                for m, mol in enumerate(mol_to_determine):
                    param['vmr_' + mol][i] = v_m_r[m]

        else:
            if param['gas_fill'] is None:
                if 'H2' in param['fit_molecules']:
                    considered_fill = 'H2'
                elif 'N2' in param['fit_molecules'] and 'H2' not in param['fit_molecules']:
                    considered_fill = 'N2'
            else:
                considered_fill = param['gas_fill']

            v_m_r = np.zeros(len(param['vmr_' + condensed_gas]))
            for mol in param['fit_molecules']:
                if mol == condensed_gas or mol == considered_fill:
                    pass
                else:
                    param['vmr_' + mol] = np.ones(len(param['vmr_' + condensed_gas])) * param['vmr_' + mol]
                    v_m_r += param['vmr_' + mol]

            param['vmr_' + considered_fill] = np.ones(len(param['vmr_' + condensed_gas])) - v_m_r - param['vmr_' + condensed_gas]

    if not param['rocky'] and (param['H2_He_ratio'] > 0):
        param['vmr_He'] = param['vmr_' + param['gas_fill']] * (1.0 - param['H2_He_ratio'])
        param['vmr_' + param['gas_fill']] *= param['H2_He_ratio']

    return param


def temp_profile(param):
    """
    Calculates temperature-pressure profile.
    Can be isothermal or based on Guillot (2010). 
    Paper ref DOI: 10.1051/0004-6361/200913396
    Paper ref link: https://www.aanda.org/articles/aa/pdf/2010/12/aa13396-09.pdf

    Parameters
    ----------
    param : dict
        dictionary of settings. Must include pressure grid, PT_profile_type 
        (isothermal, parametric, or a filepath), and (if parametric) the 
        variables for parameterization

    Returns
    -------
    T : np.array
        temperature value at each point in pressure grid.
    """
    if not isinstance(param, dict):
        raise TypeError("temp_profile expects 'param' to be a dictionary.")

    if 'P' not in param:
        raise KeyError("temp_profile requires the pressure grid under key 'P'.")

    P = np.asarray(param['P'], dtype=float)
    if P.ndim != 1:
        raise ValueError("'P' must be a one-dimensional array.")
    if P.size == 0:
        raise ValueError("'P' must contain at least one pressure value.")

    profile_type = param.get('PT_profile_type')
    if profile_type not in ('isothermal', 'parametric'):
        raise ValueError(f"Unsupported PT_profile_type '{profile_type}'.")

    if profile_type == 'isothermal':
        Tp = param.get('Tp')
        if Tp is None:
            raise ValueError("'Tp' must be provided for an isothermal PT profile.")
        try:
            Tp_val = float(Tp)
        except (TypeError, ValueError) as exc:
            raise TypeError("'Tp' must be a finite scalar.") from exc
        T = np.full(P.shape, Tp_val, dtype=float)
    else:
        required = ('kappa_th', 'gamma', 'beta', 'Tint', 'Ts', 'Rs', 'major-a', 'gp')
        missing = [key for key in required if param.get(key) is None]
        if missing:
            raise ValueError(
                "Missing parameter(s) required for a parametric PT profile: "
                + ", ".join(missing)
            )

        try:
            kappa_th = float(param['kappa_th'])
            gamma = float(param['gamma'])
            beta = float(param['beta'])
            Tint = float(param['Tint'])
            Ts = float(param['Ts'])
            Rs = float(param['Rs'])
            major_a = float(param['major-a'])
            gp = float(param['gp'])
        except (TypeError, ValueError) as exc:
            raise TypeError("Parametric PT profile parameters must be scalar numbers.") from exc

        if kappa_th <= 0.0:
            raise ValueError("'kappa_th' must be positive for a parametric PT profile.")
        if gamma <= 0.0:
            raise ValueError("'gamma' must be positive for a parametric PT profile.")
        if gp <= 0.0:
            raise ValueError("'gp' must be positive for a parametric PT profile.")
        if major_a <= 0.0:
            raise ValueError("'major-a' must be positive for a parametric PT profile.")
        if Tint < 0.0:
            raise ValueError("'Tint' must be non-negative for a parametric PT profile.")

        tau = P * kappa_th / gp
        E2 = sp.special.expn(2, gamma * tau)
        m_gamma = (
            1.0
            + (1.0 / gamma) * (1 + (0.5 * gamma * tau - 1) * np.exp(-gamma * tau))
            + gamma * (1 - 0.5 * tau**2) * E2
        )
        Teq = beta * Ts * np.sqrt(Rs * const.R_sun.value / (2 * major_a * const.au.value))
        T = (0.75 * Tint**4 * (2 / 3 + tau) + 0.5 * Teq**4 * m_gamma) ** 0.25
    return T


def ozone_earth_mask(param):
    otop, obot = (10. ** 1.5), (10. ** 4.0)
    idxs_top = np.where(otop > param['P'])[0]
    idxs_bot = np.where(param['P'] > obot)[0]
    param['vmr_O3'][idxs_top] = 10. ** (-12.0)
    param['vmr_O3'][idxs_bot] = 10. ** (-12.0)

    return param['vmr_O3']


def ranges(param):
    if param['fit_p0'] and param['gas_par_space'] != 'partial_pressure':
        param['p0_range'] = [4.5, 8.0]             # Surface pressure

    for mol in param['fit_molecules']:
        if (param['gas_par_space'] == 'centered_log_ratio' or param['gas_par_space'] == 'clr') and not param['mod_prior']:
            param['clr' + mol + '_range'] = [-25.0, 25.0]  # centered-log-ratio ranges
        elif param['gas_par_space'] == 'volume_mixing_ratio' or param['gas_par_space'] == 'vmr':
            param['vmr' + mol + '_range'] = [-12.0, 0.0]  # volume mixing ratio ranges
        elif param['gas_par_space'] == 'partial_pressure':
            param['pp' + mol + '_range'] = [-7.0, 7.0]  # partial pressure ranges

    if param['fit_ag']:
        if param['surface_albedo_parameters'] == int(1):
            param['ag_range'] = [0.0, 1.0]  # Surface albedo
        elif param['surface_albedo_parameters'] == int(3):
            for surf_alb in [1, 2]:
                param['ag' + str(surf_alb) + '_range'] = [0.0, 1.0]  # Surface albedo
            param['ag_x1_range'] = [0.4, 1.8]  # wavelength cut-off albedo
        elif param['surface_albedo_parameters'] == int(5):
            for surf_alb in [1, 2, 3]:
                param['ag' + str(surf_alb) + '_range'] = [0.0, 1.0]  # Surface albedo
            param['ag_x1_range'] = [0.4, 0.8]  # wavelength cut-off albedo
            param['ag_x2_range'] = [0.1, 1.0]

    if param['fit_T']:
        if param['PT_profile_type'] == 'isothermal':
            param['tp_range'] = [0.0, 700.0]            # Atmospheric equilibrium temperature
        elif param['PT_profile_type'] == 'parametric':
            param['kappa_th_range'] = [-10., 1.]         # thermal radiation opacity
            param['gamma_range'] = [-10., 10.]            # ratio visible opacity : thermal opacity
            param['beta_range'] = [0., 2.]             # scaling factor for equilibrium temperature (albedo)
            if param['fit_Tint']:
                param['Tint_range'] = [0., 300.]       # internal temperature

    if not param['rocky']:
        if param['fit_Rp']: 
            param['Rp_range'] = [0.1, 20.0]              # Planet radius - 0.1 to 20 Jupiter radii
        if param['fit_Mp']:
            param['Mp_range'] = [0.1, 20.0]              # Planet radius - 0.1 to 20 Jupiter masses
    else:
        if param['fit_Mp'] and param['fit_Rp']:
            if (param['Rp_prior_type'] is None or param['Rp_prior_type'] == 'independent') and (param['Mp_prior_type'] is None or param['Mp_prior_type'] == 'independent'):
                param['Mp_range'] = [0.000032, 0.06]                                     # Planet mass 0.01 to 19 Earth masses
                param['Rp_range'] = [0.044607088905052314, 0.8921417781010462]          # Planet radius - 0.5 to 10 Earth radii
            elif (param['Rp_prior_type'] is None or param['Rp_prior_type'] == 'independent') and param['Mp_prior_type'] == 'gaussian':
                param['Mp_range'] = [max(0.000032, param['Mp_orig'] - (5.0 * param['Mp_err'])), min(0.06, param['Mp_orig'] + (5.0 * param['Mp_err']))]
                param['Rp_range'] = [0.044607088905052314, 0.8921417781010462]  # Planet radius - 0.5 to 10 Earth radii
            elif param['Rp_prior_type'] == 'M_R_prior':
                if param['Mp_prior_type'] is None or param['Mp_prior_type'] == 'independent':
                    param['Mp_range'] = [0.000032, 0.06292703731012286]                   # 0.01 to 20 Earth masses
                elif param['Mp_prior_type'] == 'gaussian':
                    param['Mp_range'] = [max(0.000032, param['Mp_orig'] - (5.0 * param['Mp_err'])), min(0.06292703731012286, param['Mp_orig'] + (5.0 * param['Mp_err']))]
                elif param['Mp_prior_type'] == 'M_R_prior':
                    raise ValueError("Both Mp_prior_type and Rp_prior_type cannot be 'M_R_prior' when fitting both Mp and Rp.")
            elif param['Mp_prior_type'] == 'M_R_prior':
                if param['Rp_prior_type'] is None or param['Rp_prior_type'] == 'independent':
                    param['Rp_range'] = [0.05174422312986068, 0.19627119118223021]          # 0.58 to 2.2 Earth radii
                elif param['Rp_prior_type'] == 'gaussian':
                    param['Rp_range'] = [max(0.05174422312986068, param['Rp_orig'] - (5.0 * param['Rp_err'])), min(0.19627119118223021, param['Rp_orig'] + (5.0 * param['Rp_err']))]
                elif param['Rp_prior_type'] == 'M_R_prior':
                    raise ValueError("Both Mp_prior_type and Rp_prior_type cannot be 'M_R_prior' when fitting both Mp and Rp.")
        elif param['fit_Rp'] and param['Rp_prior_type'] == 'M_R_prior' and not param['fit_Mp'] and param['Mp'] is not None:
            param['Rp_range'] = [param['M-R_Fe'](param['Mp']), param['M-R_H2O'](param['Mp'])]
        elif param['fit_Mp'] and not param['fit_Rp']:
            param['Mp_range'] = [0.000032, 0.06]                                         # Planet mass 0.01 to 19 Earth masses
        elif param['fit_Rp'] and not param['fit_Mp']:
            param['Rp_range'] = [0.044607088905052314, 0.8921417781010462]              # Planet radius - 0.5 to 10 Earth radii
        else:
            pass

    if param['fit_g']:
        param['gp_range'] = [1.0, 6.0]  # Gravity

    if param['fit_p_size'] and param['p_size_type'] == 'constant':
        param['p_size_range'] = [-1.0, 2.0]
    elif param['fit_p_size'] and param['p_size_type'] == 'factor':
        param['p_size_range'] = [-1.0, 1.0]
    else:
        pass

    if param['fit_cld_frac']:
        param['cld_frac_range'] = [-3.0, 0.0]

    if param['fit_wtr_cld'] and param['PT_profile_type'] == 'isothermal':
        param['ptopw_range'] = [2.0, 7.0]       # Top pressure H2O
        param['dcldw_range'] = [2.0, 7.0]       # Depth H2O cloud
        param['crh2o_range'] = [-7.0, 0.0]      # Condensation Ratio H2O

    if param['fit_amm_cld'] and param['PT_profile_type'] == 'isothermal':
        param['ptopa_range'] = [2.0, 8.0]       # Top pressure NH3
        param['dclda_range'] = [2.0, 8.5]       # Depth NH3 cloud
        param['crnh3_range'] = [-7.0, 0.0]      # Condensation Ratio NH3

    if param['fit_phi']:
        if param['phi_err'] is not None:
            param['phi_range'] = [max(0.0, param['phi_orig'] - (5.0 * param['phi_err'])), min(180.0, param['phi_orig'] + (5.0 * param['phi_err']))]
        else:
            param['phi_range'] = [0.0, 180.0]       # Phase Angle

    return param


def adjust_ranges_from_dataset(param):
    """Adjust retrieval ranges to the min/max present in a dataset.

    Expects a CSV file at ``param['dataset_dir']/dataset.csv`` with a header of the form:
    ``index,<param_or_molecule>,...`` where parameter columns either match the
    molecule names in ``param['fit_molecules']`` or specific ``*_range`` keys.

    Returns the input ``param`` updated in place with new ``*_range`` bounds
    based on the dataset column-wise min/max. Raises if required columns are
    missing.
    """
    ds_dir = param.get('dataset_dir')
    if ds_dir is None:
        raise KeyError('Parameter "dataset_dir" must be set for physics_model == "dataset"')
    csv_path = os.path.join(ds_dir, 'dataset.csv')
    if not os.path.isfile(csv_path):
        raise FileNotFoundError('dataset.csv not found in: ' + ds_dir)

    # Load header and data (index | parameters...)
    with open(csv_path, 'r') as f:
        header = f.readline().strip()
    cols = [h.strip() for h in header.split(',')]
    if len(cols) < 2 or cols[0] != 'index':
        raise ValueError('Invalid dataset.csv header; first column must be "index"')

    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    X = data[:, 1:]
    colnames = cols[1:]

    # Pre-compute bounds for each column
    mins = np.nanmin(X, axis=0)
    maxs = np.nanmax(X, axis=0)
    idx_map = {name: i for i, name in enumerate(colnames)}

    def get_bounds(candidates):
        for name in candidates:
            j = idx_map.get(name)
            if j is not None:
                return [float(mins[j]), float(maxs[j])]
        return None

    # Accept alternative aliases that might appear in datasets
    alias = {
        'p0_range': ['p0_range', 'P0_range'],
        'ptopw_range': ['ptopw_range', 'Pw_top_range'],
        'dcldw_range': ['dcldw_range', 'cldw_depth_range'],
        'crh2o_range': ['crh2o_range', 'CR_H2O_range'],
        'ptopa_range': ['ptopa_range', 'Pa_top_range'],
        'dclda_range': ['dclda_range', 'clda_depth_range'],
        'crnh3_range': ['crnh3_range', 'CR_NH3_range'],
        'ag_range': ['ag_range'],
        'ag1_range': ['ag1_range'],
        'ag2_range': ['ag2_range'],
        'ag3_range': ['ag3_range'],
        'ag_x1_range': ['ag_x1_range'],
        'ag_x2_range': ['ag_x2_range'],
        'tp_range': ['tp_range', 'Tp_range'],
        'cld_frac_range': ['cld_frac_range'],
        'gp_range': ['gp_range'],
        'Mp_range': ['Mp_range'],
        'Rp_range': ['Rp_range'],
        'p_size_range': ['p_size_range', 'P_size_range'],
        'phi_range': ['phi_range']
    }

    missing = []

    # Gas parameters (one column per molecule, in log10-space)
    gps = param.get('gas_par_space')
    for mol in param['fit_molecules']:
        j = idx_map.get(mol)
        if j is None:
            missing.append(mol)
            continue
        if gps in ('volume_mixing_ratio', 'vmr'):
            param['vmr' + mol + '_range'] = [float(mins[j]), float(maxs[j])]
        elif gps == 'partial_pressure':
            param['pp' + mol + '_range'] = [float(mins[j]), float(maxs[j])]
        elif gps in ('centered_log_ratio', 'clr'):
            param['clr' + mol + '_range'] = [float(mins[j]), float(maxs[j])]

    # Scalar/free parameters mapped by *_range keys
    def require_and_set(key):
        rng = get_bounds(alias.get(key, [key]))
        if rng is None:
            missing.append(key)
        else:
            param[key] = rng

    if param['fit_p0'] and param['gas_par_space'] != 'partial_pressure':
        require_and_set('p0_range')
    if param['fit_wtr_cld']:
        require_and_set('ptopw_range')
        require_and_set('dcldw_range')
        require_and_set('crh2o_range')
    if param['fit_amm_cld']:
        require_and_set('ptopa_range')
        require_and_set('dclda_range')
        require_and_set('crnh3_range')
    if param['fit_ag']:
        if param['surface_albedo_parameters'] == int(1):
            require_and_set('ag_range')
        elif param['surface_albedo_parameters'] == int(3):
            require_and_set('ag1_range')
            require_and_set('ag2_range')
            require_and_set('ag_x1_range')
        elif param['surface_albedo_parameters'] == int(5):
            require_and_set('ag1_range')
            require_and_set('ag2_range')
            require_and_set('ag3_range')
            require_and_set('ag_x1_range')
            require_and_set('ag_x2_range')
    if param['fit_T']:
        require_and_set('tp_range')
    if param['fit_cld_frac']:
        require_and_set('cld_frac_range')
    if param['fit_g']:
        require_and_set('gp_range')
    if param['fit_Mp']:
        require_and_set('Mp_range')
    if param['fit_Rp']:
        require_and_set('Rp_range')
    if param['fit_p_size']:
        require_and_set('p_size_range')
    if param['fit_phi']:
        require_and_set('phi_range')

    if len(missing) > 0:
        raise ValueError('Required fitted parameters not found in dataset columns: ' + ', '.join(missing))

    return param


def custom_spectral_binning(x, wl, model, err=None, bins=False):
    binned_mod = []
    if err is not None:
        binned_er = []

    if not bins:
        y = np.roll(x, 1) + 0.0
        dx = (x - y)[1:]
        limits = []

        i, intermed = 0, 0
        while i in range(0, len(dx)):
            if dx[i] == dx[0]:
                lim = (dx[i] / 2., dx[i] / 2.)
                limits.append(lim)
            elif dx[i] > 2 * np.median(dx[intermed:i]):
                lim = (dx[i - 1] / 2., dx[i - 1] / 2.)
                limits.append(lim)
                i += 1
                if i != len(dx):
                    lim = (dx[i] / 2., dx[i] / 2.)
                    limits.append(lim)
                    intermed = i + 1
                else:
                    break
            else:
                lim = (dx[i - 1] / 2., dx[i] / 2.)
                limits.append(lim)
            i += 1
        limits.append((dx[i - 1] / 2., dx[i - 1] / 2.))

        for i in range(0, len(x)):
            yy = np.array(model[np.where((wl > x[i] - limits[i][0]) & (wl < x[i] + limits[i][1]))[0]])
            binned_mod.append(np.mean(yy))
            if err is not None:
                er = np.array(err[np.where((wl > x[i] - limits[i][0]) & (wl < x[i] + limits[i][1]))[0]])
                binned_er.append(np.sqrt(np.sum(er ** 2.)) / len(er))
    else:
        for i in range(0, len(x[:, 0])):
            yy = np.array(model[np.where((wl > x[i, 0]) & (wl < x[i, 1]))[0]])
            binned_mod.append(np.mean(yy))
            if err is not None:
                er = np.array(err[np.where((wl > x[i, 0]) & (wl < x[i, 1]))[0]])
                binned_er.append(np.sqrt(np.sum(er ** 2.)) / len(er))

    if err is None:
        return np.array(binned_mod)
    else:
        return np.array(binned_mod), np.array(binned_er)


def _spectres_bin_edges(wavelengths):
    """Return the bin edges inferred by SpectRes from bin centers."""
    wavelengths = np.asarray(wavelengths, dtype=float)
    if wavelengths.ndim != 1 or wavelengths.size < 2:
        raise ValueError('At least two wavelength centers are required to infer bin edges.')
    if not np.all(np.isfinite(wavelengths)) or np.any(np.diff(wavelengths) <= 0.0):
        raise ValueError('Wavelength centers must be finite and strictly increasing.')

    edges = np.empty(wavelengths.size + 1, dtype=float)
    edges[0] = wavelengths[0] - ((wavelengths[1] - wavelengths[0]) / 2.0)
    edges[-1] = wavelengths[-1] + ((wavelengths[-1] - wavelengths[-2]) / 2.0)
    edges[1:-1] = (wavelengths[1:] + wavelengths[:-1]) / 2.0
    return edges


def _spectres_integrated_flux(wavelength, flux_density, wl_low, wl_high):
    """Integrate a flux density over explicit bins using SpectRes.

    SpectRes normally infers output-bin edges from output wavelength centers.
    For every contiguous group of requested bins, this routine constructs the
    SpectRes wavelength basis whose inferred edges are exactly ``wl_low`` and
    ``wl_high``. The returned bin-averaged flux density is then multiplied by
    the actual bin width, yielding integrated flux.

    Parameters are expected in compatible wavelength units. If ``flux_density``
    is W m-2 micron-1 and wavelengths are micron, the output is W m-2.
    """
    wavelength = np.asarray(wavelength, dtype=float)
    flux_density = np.asarray(flux_density, dtype=float)
    wl_low = np.asarray(wl_low, dtype=float)
    wl_high = np.asarray(wl_high, dtype=float)

    if wavelength.ndim != 1 or flux_density.shape != wavelength.shape:
        raise ValueError('Stellar wavelength and flux-density arrays must be one-dimensional and have matching shapes.')
    if wl_low.ndim != 1 or wl_high.shape != wl_low.shape or wl_low.size == 0:
        raise ValueError('Observation bin limits must be non-empty, one-dimensional arrays with matching shapes.')
    if not np.all(np.isfinite(wavelength)) or not np.all(np.isfinite(flux_density)):
        raise ValueError('Stellar wavelength and flux-density arrays must contain only finite values.')
    if np.any(np.diff(wavelength) <= 0.0):
        raise ValueError('Stellar wavelengths must be strictly increasing.')
    if not np.all(np.isfinite(wl_low)) or not np.all(np.isfinite(wl_high)):
        raise ValueError('Observation bin limits must contain only finite values.')
    if np.any(wl_high <= wl_low):
        raise ValueError('Every observation bin must have wl_high > wl_low.')
    bin_centers = (wl_low + wl_high) / 2.0
    if np.any(np.diff(bin_centers) <= 0.0):
        raise ValueError('Observation wavelength bins must be ordered by increasing center.')

    integrated_flux = np.empty(wl_low.size, dtype=float)

    def resample_group(start, stop):
        edges = np.concatenate(([wl_low[start]], wl_high[start:stop]))
        spectres_wavelengths = np.empty(edges.size, dtype=float)
        spectres_wavelengths[0] = (edges[0] + edges[1]) / 2.0
        for edge_idx in range(1, edges.size):
            spectres_wavelengths[edge_idx] = (2.0 * edges[edge_idx]) - spectres_wavelengths[edge_idx - 1]

        # Strongly varying adjacent bin widths can make the constructed basis
        # non-monotonic. In that uncommon case, process each bin independently.
        if np.any(np.diff(spectres_wavelengths) <= 0.0):
            for bin_idx in range(start, stop):
                center = (wl_low[bin_idx] + wl_high[bin_idx]) / 2.0
                width = wl_high[bin_idx] - wl_low[bin_idx]
                basis = np.array([center, center + width])
                mean_flux_density = spectres(
                    basis, wavelength, flux_density, fill=np.nan, verbose=False
                )[0]
                integrated_flux[bin_idx] = mean_flux_density * width
            return

        mean_flux_density = spectres(
            spectres_wavelengths, wavelength, flux_density, fill=np.nan, verbose=False
        )
        integrated_flux[start:stop] = mean_flux_density[:-1] * (wl_high[start:stop] - wl_low[start:stop])

    group_start = 0
    for bin_idx in range(wl_low.size - 1):
        # Gaps and overlaps both terminate a contiguous SpectRes group. This
        # permits independently defined channels from adjacent instruments to
        # overlap while still integrating every channel over its exact edges.
        contiguous = np.isclose(wl_high[bin_idx], wl_low[bin_idx + 1], rtol=1.0e-10, atol=1.0e-12)
        if not contiguous:
            resample_group(group_start, bin_idx + 1)
            group_start = bin_idx + 1
    resample_group(group_start, wl_low.size)

    if not np.all(np.isfinite(integrated_flux)):
        raise ValueError('The PHOENIX spectrum does not fully cover the requested observation bins.')
    return integrated_flux


def _star_spectrum_entry(param, n_obs=None):
    """Return the stellar spectrum corresponding to one observation."""
    expected_signature = _stellar_model_signature(param)
    if (
        'starfx' not in param
        or tuple(param.get('starfx_model', {}).get('model_signature', ()))
        != expected_signature
    ):
        take_star_spectrum(param)
    if param['obs_numb'] is None:
        return param['starfx']
    obs = 0 if n_obs is None else int(n_obs)
    return param['starfx'][str(obs)]


def _opacity_resolving_power(param):
    """Return the nominal molecular-opacity resolving power when available."""
    resolution_label = param.get('opac_data')
    if isinstance(resolution_label, (int, float, np.integer, np.floating)):
        return float(resolution_label)
    if isinstance(resolution_label, str):
        match = re.search(r'(?i)(\d+(?:\.\d+)?)\s*k', resolution_label)
        if match:
            return float(match.group(1)) * 1000.0
        match = re.fullmatch(r'(?i)\s*r?\s*(\d+(?:\.\d+)?)\s*', resolution_label)
        if match:
            return float(match.group(1))
        match = re.search(
            r'(?i)(?:^|[_-])r(\d+(?:\.\d+)?)(?:$|[_-])', resolution_label
        )
        if match:
            return float(match.group(1))

    wavelength = param.get('opacw')
    if wavelength is None:
        return None
    wavelength = np.asarray(wavelength, dtype=float).reshape(-1)
    wavelength = wavelength[np.isfinite(wavelength) & (wavelength > 0.0)]
    if wavelength.size < 2:
        return None
    wavelength = np.sort(np.unique(wavelength))
    delta = np.diff(wavelength)
    resolving_power = wavelength[:-1] / delta
    resolving_power = resolving_power[np.isfinite(resolving_power) & (resolving_power > 0.0)]
    if resolving_power.size == 0:
        return None
    return float(np.median(resolving_power))


def _warn_if_stellar_grid_is_too_coarse(param, stellar_spectrum_required=False):
    """Warn when high-resolution opacities are paired with the standard atlas."""
    resolving_power = _opacity_resolving_power(param)
    if resolving_power is not None:
        param['opacity_resolving_power'] = resolving_power

    if (
        resolving_power is None
        or resolving_power <= 10000.0
        or (
            not stellar_spectrum_required
            and (param.get('albedo_calc', False) or param.get('fp_over_fs', False))
        )
        or param.get('stellar_dir') is not None
        or param.get('_stellar_resolution_warning_emitted', False)
    ):
        return

    message = (
        f"Molecular opacity resolving power R={resolving_power:g} exceeds 10,000, "
        "while the standard stsynphot PHOENIX atlas is being used. Consider "
        "providing higher-resolution SVO stellar spectra with 'stellar_dir'."
    )
    # __basics.py suppresses third-party warnings globally; force this ExoReL
    # science warning to remain visible without turning it into an exception.
    with warnings.catch_warnings():
        warnings.simplefilter('always', RuntimeWarning)
        warnings.warn(message, RuntimeWarning, stacklevel=3)
    param['_stellar_resolution_warning_emitted'] = True


def _read_svo_stellar_header(path):
    """Read stellar parameters and declared units from an SVO ASCII header."""
    metadata = {}
    patterns = {
        'temperature_K': re.compile(r'^\s*#\s*teff\s*=\s*([+-]?[\d.]+(?:[eEdD][+-]?\d+)?)', re.I),
        'logg_cgs': re.compile(r'^\s*#\s*logg\s*=\s*([+-]?[\d.]+(?:[eEdD][+-]?\d+)?)', re.I),
        'metallicity': re.compile(r'^\s*#\s*meta\s*=\s*([+-]?[\d.]+(?:[eEdD][+-]?\d+)?)', re.I),
    }
    unit_patterns = {
        'wavelength_unit': re.compile(
            r'^\s*#\s*column\s+1\s*:\s*wavelength\s*\(\s*angstrom\s*\)', re.I
        ),
        'flux_density_unit': re.compile(
            r'^\s*#\s*column\s+2\s*:\s*flux\s*'
            r'\(\s*erg\s*/\s*cm2\s*/\s*s\s*/\s*a\s*\)', re.I
        ),
    }
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as stream:
            for _ in range(40):
                line = stream.readline()
                if not line or not line.lstrip().startswith('#'):
                    break
                for key, pattern in patterns.items():
                    match = pattern.search(line)
                    if match:
                        metadata[key] = float(match.group(1).replace('D', 'E').replace('d', 'e'))
                for key, pattern in unit_patterns.items():
                    if pattern.search(line):
                        metadata[key] = (
                            'angstrom' if key == 'wavelength_unit'
                            else 'erg cm-2 s-1 angstrom-1'
                        )
    except OSError:
        return None
    if not set(patterns).issubset(metadata):
        return None
    metadata.setdefault('wavelength_unit', None)
    metadata.setdefault('flux_density_unit', None)
    # The SVO Theory Server distributes these BT-Settl coordinates in air.
    # ExoReL converts the optical part back to vacuum when loading the data.
    metadata['input_wavelength_medium'] = 'air'
    metadata['path'] = path
    return metadata


def _svo_interpolation_models(stellar_dir, temperature, metallicity, logg, wkg_dir):
    """Return the SVO grid models and trilinear weights for a target star."""
    stellar_dir = os.path.expanduser(os.fspath(stellar_dir))
    if not os.path.isabs(stellar_dir):
        stellar_dir = os.path.join(wkg_dir, stellar_dir)
    stellar_dir = os.path.abspath(stellar_dir)
    if not os.path.isdir(stellar_dir):
        raise FileNotFoundError(f"The stellar_dir directory does not exist: {stellar_dir}")

    supported_suffixes = ('.txt', '.dat', '.spec', '.7')
    candidates = []
    with os.scandir(stellar_dir) as entries:
        for entry in sorted(entries, key=lambda item: item.name):
            if not entry.is_file() or not entry.name.lower().endswith(supported_suffixes):
                continue
            metadata = _read_svo_stellar_header(entry.path)
            if metadata is not None:
                candidates.append(metadata)
    if not candidates:
        raise FileNotFoundError(
            f"No SVO stellar ASCII spectra were found in {stellar_dir}. Expected header "
            "entries for teff, logg, and meta."
        )

    parameter_targets = {
        'temperature_K': temperature,
        'logg_cgs': logg,
        'metallicity': metallicity,
    }

    def bracket(key, target):
        values = np.array(sorted({model[key] for model in candidates}), dtype=float)
        exact = values[np.isclose(values, target, rtol=0.0, atol=1.0e-10)]
        if exact.size:
            return [(float(exact[0]), 1.0)]
        lower = values[values < target]
        upper = values[values > target]
        if lower.size == 0 or upper.size == 0:
            raise ValueError(
                f"Requested stellar {key}={target:g} is outside the SVO grid "
                f"range [{values.min():g}, {values.max():g}] in {stellar_dir}."
            )
        low = float(lower.max())
        high = float(upper.min())
        return [
            (low, (high - target) / (high - low)),
            (high, (target - low) / (high - low)),
        ]

    corners = [({}, 1.0)]
    for key, target in parameter_targets.items():
        corners = [
            ({**corner, key: value}, weight * axis_weight)
            for corner, weight in corners
            for value, axis_weight in bracket(key, target)
        ]

    interpolation_models = []
    missing = []
    for corner, weight in corners:
        matches = [
            model for model in candidates
            if all(np.isclose(model[key], value, rtol=0.0, atol=1.0e-10)
                   for key, value in corner.items())
        ]
        if not matches:
            missing.append(corner)
            continue
        if len(matches) > 1:
            paths = ', '.join(model['path'] for model in matches)
            raise ValueError(
                "Multiple SVO spectra have identical stellar parameters; keep only "
                f"one of these files in stellar_dir: {paths}"
            )
        model = dict(matches[0])
        model['weight'] = float(weight)
        interpolation_models.append(model)

    if missing:
        missing_text = '; '.join(
            f"Teff={corner['temperature_K']:g} K, logg={corner['logg_cgs']:g}, "
            f"[M/H]={corner['metallicity']:g}"
            for corner in missing
        )
        raise ValueError(
            "stellar_dir does not contain the complete bounding model set needed "
            f"for trilinear interpolation. Missing: {missing_text}"
        )

    return interpolation_models


def _svo_air_to_vacuum_wavelength(wavelength_air):
    """Convert SVO optical air wavelengths to vacuum using Morton (2000).

    Wavelengths below 2000 Angstrom are conventionally vacuum wavelengths and
    are left unchanged. The second return value is d(lambda_air)/d(lambda_vac)
    and is used to transform a per-air-wavelength flux density conservatively.
    """
    wavelength_air = np.asarray(wavelength_air, dtype=np.float64)
    wavelength_vacuum = wavelength_air.copy()
    density_jacobian = np.ones_like(wavelength_air)
    optical = wavelength_air >= 2000.0
    if not np.any(optical):
        return wavelength_vacuum, density_jacobian

    air = wavelength_air[optical]
    vacuum = air.copy()
    for _ in range(5):
        sigma_squared = (1.0e4 / vacuum) ** 2.0
        refractive_index = (
            1.0
            + 8.34254e-5
            + (2.406147e-2 / (130.0 - sigma_squared))
            + (1.5998e-4 / (38.9 - sigma_squared))
        )
        vacuum = air * refractive_index

    sigma_squared = (1.0e4 / vacuum) ** 2.0
    refractive_index = (
        1.0
        + 8.34254e-5
        + (2.406147e-2 / (130.0 - sigma_squared))
        + (1.5998e-4 / (38.9 - sigma_squared))
    )
    dsigma_squared_dvacuum = -2.0 * sigma_squared / vacuum
    dn_dvacuum = (
        (2.406147e-2 * dsigma_squared_dvacuum / (130.0 - sigma_squared) ** 2.0)
        + (1.5998e-4 * dsigma_squared_dvacuum / (38.9 - sigma_squared) ** 2.0)
    )
    wavelength_vacuum[optical] = vacuum
    density_jacobian[optical] = (
        (refractive_index - (vacuum * dn_dvacuum)) / refractive_index ** 2.0
    )
    return wavelength_vacuum, density_jacobian


def _load_svo_stellar_spectrum(model, SourceSpectrum, Empirical1D, u):
    """Load an SVO spectrum with validated units on a vacuum wavelength grid."""
    if (
        model.get('wavelength_unit') != 'angstrom'
        or model.get('flux_density_unit') != 'erg cm-2 s-1 angstrom-1'
    ):
        raise ValueError(
            "Unsupported or missing SVO column units in "
            f"{model['path']}. Expected WAVELENGTH (ANGSTROM) and "
            "FLUX (ERG/CM2/S/A)."
        )
    try:
        # Parse in float64 even when returned ExoReL arrays use float32. Casting
        # the native wavelength grid first would merge genuinely distinct
        # high-resolution samples at long wavelengths.
        data = np.loadtxt(model['path'], comments='#', usecols=(0, 1), dtype=np.float64)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read SVO stellar spectrum: {model['path']}") from exc
    data = np.atleast_2d(data)
    if data.shape[0] < 2 or data.shape[1] != 2:
        raise ValueError(f"SVO stellar spectrum has insufficient data: {model['path']}")

    wavelength = np.asarray(data[:, 0], dtype=np.float64)
    flux_density = np.asarray(data[:, 1], dtype=np.float64)
    valid = np.isfinite(wavelength) & np.isfinite(flux_density) & (wavelength > 0.0)
    wavelength = wavelength[valid]
    flux_density = flux_density[valid]
    if wavelength.size < 2:
        raise ValueError(f"SVO stellar spectrum has insufficient finite data: {model['path']}")

    order = np.argsort(wavelength, kind='stable')
    wavelength = wavelength[order]
    flux_density = flux_density[order]
    unique_wavelength, first_index, counts = np.unique(
        wavelength, return_index=True, return_counts=True
    )
    duplicate_count = int(wavelength.size - unique_wavelength.size)
    if duplicate_count:
        summed_flux = np.add.reduceat(flux_density.astype(np.float64), first_index)
        flux_density = summed_flux / counts
        wavelength = unique_wavelength

    wavelength, density_jacobian = _svo_air_to_vacuum_wavelength(wavelength)
    flux_density = flux_density * density_jacobian
    stellar_spectrum = SourceSpectrum(
        Empirical1D,
        points=wavelength * u.AA,
        lookup_table=flux_density * (u.erg / (u.cm ** 2.0 * u.s * u.AA)),
        keep_neg=True,
    )
    return stellar_spectrum, duplicate_count


def _interpolate_svo_stellar_spectra(models, SourceSpectrum, Empirical1D, u):
    """Combine SVO spectra using their trilinear parameter-space weights."""
    interpolated_spectrum = None
    source_models = []
    for model in models:
        spectrum, duplicate_count = _load_svo_stellar_spectrum(
            model, SourceSpectrum, Empirical1D, u
        )
        weighted_spectrum = model['weight'] * spectrum
        if interpolated_spectrum is None:
            interpolated_spectrum = weighted_spectrum
        else:
            interpolated_spectrum = interpolated_spectrum + weighted_spectrum
        source_models.append({
            'file': model['path'],
            'temperature_K': model['temperature_K'],
            'metallicity': model['metallicity'],
            'logg_cgs': model['logg_cgs'],
            'weight': model['weight'],
            'duplicate_wavelengths_collapsed': duplicate_count,
            'input_wavelength_unit': model['wavelength_unit'],
            'input_flux_density_unit': model['flux_density_unit'],
            'input_wavelength_medium': model['input_wavelength_medium'],
            'working_wavelength_medium': 'vacuum',
            'air_to_vacuum_method': 'Morton (2000); optical wavelengths >= 2000 Angstrom',
        })

    if interpolated_spectrum is None:
        raise RuntimeError('No SVO stellar spectra were available for interpolation.')
    return interpolated_spectrum, source_models


def model_finalizzation(param, alb_wl, alb, planet_albedo=False, fp_over_fs=False, n_obs=None):
    if not param['wl_native']:
        if param['obs_numb'] is not None:
            observation = param['spectrum'][str(n_obs)]
        else:
            observation = param['spectrum']
        wl = observation['wl'] + 0.0

        if param['spectrum']['bins']:
            wl_bins = np.column_stack((observation['wl_low'], observation['wl_high']))
            albedo = custom_spectral_binning(wl_bins, alb_wl, alb, bins=True)
        else:
            albedo = spectres(wl, alb_wl, alb, fill=False)

        wl_i = find_nearest(wl, param['min_wl'] - 0.05)
        wl_f = find_nearest(wl, param['max_wl'] + 0.05)
        wl = wl[wl_i: wl_f + 1]
        albedo = albedo[wl_i: wl_f + 1]
    else:
        wl = alb_wl + 0.0
        albedo = alb + 0.0

    if param['flat_albedo']:
        albedo = np.ones(len(albedo)) * param['flat_albedo_value']

    if planet_albedo and not fp_over_fs:
        return wl, albedo
    elif fp_over_fs and not planet_albedo:
        contrast = albedo * (((param['Rp'] * const.R_jup.value) / (param['major-a'] * const.au.value)) ** 2.0)
        return wl, contrast
    elif not planet_albedo and not fp_over_fs:
        contrast = albedo * (((param['Rp'] * const.R_jup.value) / (param['major-a'] * const.au.value)) ** 2.0)
        star_flux = _star_spectrum_entry(param, n_obs=n_obs)['y']
        planet_flux = contrast * star_flux * (((param['Rs'] * const.R_sun.value) / (param['distance'] * const.pc.value)) ** 2.0)
        return wl, planet_flux


def _stellar_model_signature(param):
    """Return current stellar-grid parameters, deriving gravity when needed."""
    temperature = float(param['Ts'])
    metallicity = 0.0 if param.get('meta') is None else float(param['meta'])
    if param.get('Loggs') is None or param.get('_stellar_logg_derived', False):
        if param.get('Ms') is None or param.get('Rs') is None:
            raise ValueError('Provide Loggs, or both Ms and Rs, to select a PHOENIX stellar model.')
        gravity_cgs = (
            const.G.value
            * (float(param['Ms']) * const.M_sun.value)
            / ((float(param['Rs']) * const.R_sun.value) ** 2.0)
            * 100.0
        )
        param['Loggs'] = float(np.log10(gravity_cgs))
        param['_stellar_logg_derived'] = True
    logg = float(param['Loggs'])

    stellar_dir = param.get('stellar_dir')
    if stellar_dir is not None:
        stellar_dir = os.path.expanduser(os.fspath(stellar_dir))
        if not os.path.isabs(stellar_dir):
            stellar_dir = os.path.join(param.get('wkg_dir', os.getcwd()), stellar_dir)
        stellar_dir = os.path.abspath(stellar_dir)
    return temperature, metallicity, logg, stellar_dir


def _build_stellar_source_spectrum(param):
    """Return the selected PHOENIX surface spectrum and model metadata."""
    try:
        import stsynphot as stsyn
        from astropy import units as u
        from synphot import SourceSpectrum
        from synphot.models import Empirical1D
    except ImportError as exc:
        raise ImportError(
            "Stellar-spectrum calculations require 'stsynphot' and 'synphot'. Install "
            "the project requirements and retry."
        ) from exc

    configured_cdbs = os.environ.get('PYSYN_CDBS')
    if configured_cdbs:
        stsyn.conf.rootdir = configured_cdbs

    cache_signature = _stellar_model_signature(param)
    temperature, metallicity, logg, stellar_dir = cache_signature
    cached = param.get('_stellar_source_cache')
    if cached is not None and cached.get('signature') == cache_signature:
        return cached['spectrum'], copy.deepcopy(cached['metadata'])

    if stellar_dir is not None:
        interpolation_models = _svo_interpolation_models(
            stellar_dir,
            temperature,
            metallicity,
            logg,
            param.get('wkg_dir', os.getcwd()),
        )
        stellar_spectrum, source_models = _interpolate_svo_stellar_spectra(
            interpolation_models, SourceSpectrum, Empirical1D, u
        )
        if not param.get('_stellar_source_message_emitted', False):
            if len(source_models) == 1:
                external_message = (
                    "ExoReL: Using external high-resolution stellar spectrum "
                    f"'{source_models[0]['file']}'"
                )
            else:
                external_message = (
                    "ExoReL: Using trilinear interpolation of "
                    f"{len(source_models)} external high-resolution stellar spectra"
                )
            duplicate_count = sum(
                model['duplicate_wavelengths_collapsed'] for model in source_models
            )
            print(
                external_message
                + f" for Teff={temperature:g} K, logg={logg:g}, [M/H]={metallicity:g}."
                + (
                    f" Collapsed {duplicate_count} rounded duplicate wavelengths "
                    "across the source files."
                    if duplicate_count else ""
                ),
                flush=True,
            )
            param['_stellar_source_message_emitted'] = True
        model_metadata = {
            'library': 'synphot',
            'grid': 'external_svo_bt-settl',
            'interpolation': 'exact' if len(source_models) == 1 else 'trilinear',
            'temperature_K': temperature,
            'metallicity': metallicity,
            'logg_cgs': logg,
            'source_models': source_models,
        }
    else:
        _warn_if_stellar_grid_is_too_coarse(param, stellar_spectrum_required=True)
        try:
            stellar_spectrum = stsyn.grid_to_spec('phoenix', temperature, metallicity, logg)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "The configured stsynphot PHOENIX reference atlas was not found. "
                "Run ExoReL.setup_stsynphot_data() or set PYSYN_CDBS to an STScI "
                "reference-atlas directory containing grid/phoenix."
            ) from exc
        model_metadata = {
            'library': 'stsynphot',
            'grid': 'phoenix',
            'temperature_K': temperature,
            'metallicity': metallicity,
            'logg_cgs': logg,
        }

    param['_stellar_source_cache'] = {
        'signature': cache_signature,
        'spectrum': stellar_spectrum,
        'metadata': copy.deepcopy(model_metadata),
    }
    return stellar_spectrum, model_metadata


def _prepare_python_stellar_irradiance(param):
    """Evaluate and dilute the selected stellar flux density on the opacity grid."""
    try:
        from astropy import units as u
    except ImportError as exc:
        raise ImportError(
            "Python stellar irradiance requires 'astropy'. Install the project "
            "requirements and retry."
        ) from exc

    if 'opacw' not in param:
        raise RuntimeError('The opacity wavelength grid must be loaded before stellar irradiance.')
    if param.get('Rs') is None or param.get('major-a') is None:
        raise ValueError('Rs and major-a are required to dilute stellar flux to the planet.')

    target_dtype = np.float32 if param.get('use_float32', False) else np.float64
    wavelength_nm = np.asarray(param['opacw'], dtype=np.float64).reshape(-1) * 1.0e9
    wavelength_micron = wavelength_nm * 1.0e-3
    model_signature = _stellar_model_signature(param)
    existing_irradiance = param.get('stellar_irradiance')
    existing_wavelength = None
    if existing_irradiance is not None:
        existing_wavelength = np.asarray(
            existing_irradiance.get('wavelength_nm', []), dtype=np.float64
        ).reshape(-1)
    reuse_surface_flux = (
        existing_irradiance is not None
        and tuple(existing_irradiance.get('model_signature', ())) == model_signature
        and existing_wavelength.shape == wavelength_nm.shape
        and np.allclose(existing_wavelength, wavelength_nm, rtol=5.0e-7, atol=1.0e-6)
    )

    if reuse_surface_flux:
        surface_flux_per_nm = np.asarray(
            existing_irradiance['surface_flux_density'], dtype=np.float64
        )
        model_metadata = copy.deepcopy(param['stellar_irradiance_model'])
    else:
        stellar_spectrum, model_metadata = _build_stellar_source_spectrum(param)
        surface_flux_per_micron = np.asarray(
            stellar_spectrum(
                wavelength_micron * u.micron,
                flux_unit=u.W / (u.m ** 2.0 * u.micron),
            ).value,
            dtype=np.float64,
        )
        # The Python radiative-transfer source and Planck function use spectral
        # densities per nm. 1 micron contains 1000 nm.
        surface_flux_per_nm = surface_flux_per_micron * 1.0e-3

    stellar_radius_m = float(param['Rs']) * const.R_sun.value
    orbital_distance_m = float(param['major-a']) * const.au.value
    if stellar_radius_m <= 0.0 or orbital_distance_m <= 0.0:
        raise ValueError('Rs and major-a must be positive to dilute stellar flux.')
    dilution_factor = (stellar_radius_m / orbital_distance_m) ** 2.0
    planet_flux_per_nm = surface_flux_per_nm * dilution_factor

    if (
        surface_flux_per_nm.shape != wavelength_nm.shape
        or not np.all(np.isfinite(surface_flux_per_nm))
        or np.any(surface_flux_per_nm <= 0.0)
    ):
        raise ValueError(
            'The selected PHOENIX spectrum does not provide positive, finite flux '
            'density across the opacity wavelength grid.'
        )

    param['stellar_irradiance'] = {
        'wavelength_nm': np.asarray(wavelength_nm, dtype=target_dtype),
        'surface_flux_density': np.asarray(surface_flux_per_nm, dtype=target_dtype),
        'planet_flux_density': np.asarray(planet_flux_per_nm, dtype=target_dtype),
        'wavelength_unit': 'nm',
        'flux_density_unit': 'W m-2 nm-1',
        'dilution_factor': float(dilution_factor),
        'stellar_radius_Rsun': float(param['Rs']),
        'orbital_distance_au': float(param['major-a']),
        'model_signature': model_signature,
    }
    irradiance_metadata = copy.deepcopy(model_metadata)
    irradiance_metadata.update({
        'evaluation_grid': 'molecular-opacity wavelength grid',
        'surface_flux_density_unit': 'W m-2 nm-1',
        'planet_flux_density_unit': 'W m-2 nm-1',
        'dilution': '(Rs * R_sun / (major-a * au))^2',
        'dilution_factor': float(dilution_factor),
        'dtype': np.dtype(target_dtype).name,
        'model_signature': model_signature,
    })
    param['stellar_irradiance_model'] = irradiance_metadata
    param.pop('_stellar_source_cache', None)
    return param


def take_star_spectrum(param, plot=False):
    try:
        from astropy import units as u
    except ImportError as exc:
        raise ImportError(
            "take_star_spectrum requires 'astropy'. Install the project requirements "
            "and retry."
        ) from exc

    stellar_spectrum, model_metadata = _build_stellar_source_spectrum(param)
    temperature = float(model_metadata['temperature_K'])
    logg = float(model_metadata['logg_cgs'])
    target_dtype = np.float32 if param.get('use_float32', False) else np.float64

    native_wavelength = np.asarray(
        stellar_spectrum.waveset.to_value(u.micron), dtype=target_dtype
    )
    native_flux_density = np.asarray(stellar_spectrum(
        stellar_spectrum.waveset,
        flux_unit=u.W / (u.m ** 2.0 * u.micron),
    ).value, dtype=target_dtype)

    def rebin_for_observation(observation):
        output_wavelength = np.asarray(observation['wl'], dtype=target_dtype)
        if observation.get('wl_low') is not None and observation.get('wl_high') is not None:
            wl_low = np.asarray(observation['wl_low'], dtype=target_dtype)
            wl_high = np.asarray(observation['wl_high'], dtype=target_dtype)
            integrated_flux = _spectres_integrated_flux(
                native_wavelength, native_flux_density, wl_low, wl_high
            )
        else:
            edges = _spectres_bin_edges(output_wavelength)
            mean_flux_density = spectres(
                output_wavelength,
                native_wavelength,
                native_flux_density,
                fill=np.nan,
                verbose=False,
            )
            integrated_flux = mean_flux_density * np.diff(edges)

        integrated_flux = np.asarray(integrated_flux, dtype=target_dtype)

        if output_wavelength.shape != integrated_flux.shape:
            raise RuntimeError('Stellar wavelength and integrated-flux arrays have inconsistent shapes.')
        if not np.all(np.isfinite(integrated_flux)):
            raise ValueError('The PHOENIX spectrum does not fully cover the observation wavelength grid.')

        return {
            'x': output_wavelength,
            'y': integrated_flux,
            'x_unit': 'micron',
            'y_unit': 'W m-2',
        }

    if param['obs_numb'] is None:
        param['starfx'] = rebin_for_observation(param['spectrum'])
        plot_entries = [param['starfx']]
    else:
        param['starfx'] = {}
        for obs in range(0, int(param['obs_numb'])):
            param['starfx'][str(obs)] = rebin_for_observation(param['spectrum'][str(obs)])
        plot_entries = [param['starfx'][str(obs)] for obs in range(0, int(param['obs_numb']))]

    model_metadata['dtype'] = np.dtype(target_dtype).name
    model_metadata['model_signature'] = _stellar_model_signature(param)
    param['starfx_model'] = model_metadata
    if 'stellar_irradiance' in param:
        param.pop('_stellar_source_cache', None)

    if plot:
        for entry in plot_entries:
            plt.plot(entry['x'], entry['y'], '-')
        plt.grid()
        plt.xlabel(r'Wavelength ($\mu$m)')
        plt.ylabel('Integrated stellar surface flux per bin (W/m$^2$)')
        plt.title(f'PHOENIX stellar spectrum: T={temperature:g} K, log(g)={logg:.2f}')
        plot_dir = os.path.join(param['wkg_dir'], 'Retrieval')
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, 'Star_spectrum.pdf'))
        plt.close()

    return param


def pre_load_variables(param):
    if param['physics_model_code_language'] == 'Python':
        zone_data = np.genfromtxt(param['pkg_dir'] + 'forward_mod/Data/zone_Earth_Full.dat', dtype=str, skip_header=1)
        if zone_data.ndim == 1 and zone_data.size:
            zone_data = zone_data.reshape(1, -1)
        param['zone_data'] = zone_data
        param = load_reactions(param)
        param = load_photolysis(param)
        param = load_cross(param)
        param = _prepare_python_stellar_irradiance(param)
        param = load_cia(param)
        param['aer_h2so4'] = np.loadtxt(param['pkg_dir'] + 'forward_mod/Data/H2SO4AER_CrossM_01.dat')
        param['aer_s8'] = np.loadtxt(param['pkg_dir'] + 'forward_mod/Data/S8AER_CrossM_01.dat')

    if param['rocky']:
    # Mass-Radius diagram
        if param['fit_Mp'] or param['fit_Rp']:
            M_R_Fe = np.loadtxt(param['pkg_dir'] + 'forward_mod/Data/Fe_mass_radius_jup.dat')
            M_R_H2O = np.loadtxt(param['pkg_dir'] + 'forward_mod/Data/H2O_mass_radius_jup.dat')
            if param['Rp_prior_type'] == 'M_R_prior' and param['Mp_prior_type'] != 'M_R_prior':
                param['M-R_Fe'] = interp1d(M_R_Fe[:, 0], M_R_Fe[:, 1])
                param['M-R_H2O'] = interp1d(M_R_H2O[:, 0], M_R_H2O[:, 1])
            elif param['Mp_prior_type'] == 'M_R_prior' and param['Rp_prior_type'] != 'M_R_prior':
                param['M-R_Fe'] = interp1d(M_R_Fe[:, 1], M_R_Fe[:, 0])
                param['M-R_H2O'] = interp1d(M_R_H2O[:, 1], M_R_H2O[:, 0])
            else:
                pass
    #  Load Mie Calculation Results
    fldr_cld_fl = 'forward_mod/cloud_files/'
    if param['fit_wtr_cld']:
        if param['wtr_cld_type'] == 'liquid':
            data = np.loadtxt(param['pkg_dir'] + fldr_cld_fl + 'Cross_water_wavelength_250916.dat')
            param['H2OL_r'] = data[:, 0]  # zero-order radius, in micron
            param['H2OL_c'] = data[:, 1:]  # cross-section per droplet, in cm2
            data = np.loadtxt(param['pkg_dir'] + fldr_cld_fl + 'Albedo_water_wavelength_250916.dat')
            param['H2OL_a'] = data[:, 1:]
            data = np.loadtxt(param['pkg_dir'] + fldr_cld_fl + 'Geo_water_wavelength_250916.dat')
            param['H2OL_g'] = data[:, 1:]
        elif param['wtr_cld_type'] == 'ice':
            data = np.loadtxt(param['pkg_dir'] + fldr_cld_fl + 'Cross_ice_wavelength_250916.dat')
            param['H2OL_r'] = data[:, 0]  # zero-order radius, in micron
            param['H2OL_c'] = data[:, 1:]  # cross-section per droplet, in cm2
            data = np.loadtxt(param['pkg_dir'] + fldr_cld_fl + 'Albedo_ice_wavelength_250916.dat')
            param['H2OL_a'] = data[:, 1:]
            data = np.loadtxt(param['pkg_dir'] + fldr_cld_fl + 'Geo_ice_wavelength_250916.dat')
            param['H2OL_g'] = data[:, 1:]
        elif param['wtr_cld_type'] == 'mixed' and param['PT_profile_type'] == 'parametric':
            data = np.loadtxt(param['pkg_dir'] + fldr_cld_fl + 'Cross_water_wavelength_250916.dat')
            param['H2OL_r'] = data[:, 0]  # zero-order radius, in micron
            param['H2OL_c_liquid'] = data[:, 1:]  # cross-section per droplet, in cm2
            data = np.loadtxt(param['pkg_dir'] + fldr_cld_fl + 'Cross_ice_wavelength_250916.dat')
            param['H2OL_c_ice'] = data[:, 1:]  # cross-section per droplet, in cm2

            data = np.loadtxt(param['pkg_dir'] + fldr_cld_fl + 'Albedo_water_wavelength_250916.dat')
            param['H2OL_a_liquid'] = data[:, 1:]
            data = np.loadtxt(param['pkg_dir'] + fldr_cld_fl + 'Albedo_ice_wavelength_250916.dat')
            param['H2OL_a_ice'] = data[:, 1:]

            data = np.loadtxt(param['pkg_dir'] + fldr_cld_fl + 'Geo_water_wavelength_250916.dat')
            param['H2OL_g_liquid'] = data[:, 1:]
            data = np.loadtxt(param['pkg_dir'] + fldr_cld_fl + 'Geo_ice_wavelength_250916.dat')
            param['H2OL_g_ice'] = data[:, 1:]
    
    if param['fit_amm_cld']:
        data = np.loadtxt(param['pkg_dir'] + fldr_cld_fl + 'Cross_ammonia_wavelength_250916.dat')
        param['NH3_r'] = data[:, 0]  # zero-order radius, in micron
        param['NH3_c'] = data[:, 1:]  # cross-section per droplet, in cm2
        data = np.loadtxt(param['pkg_dir'] + fldr_cld_fl + 'Albedo_ammonia_wavelength_250916.dat')
        param['NH3_a'] = data[:, 1:]
        data = np.loadtxt(param['pkg_dir'] + fldr_cld_fl + 'Geo_ammonia_wavelength_250916.dat')
        param['NH3_g'] = data[:, 1:]

    return param


def load_reactions(param):
    def _load_reaction_table(path, width):
        data = np.loadtxt(path, dtype=int)
        data = np.asarray(data, dtype=int)
        if data.size == 0:
            return np.zeros((1, width + 1), dtype=int)
        data = np.atleast_2d(data)
        table = np.zeros((data.shape[0] + 1, width + 1), dtype=int)
        table[1:, 1:width + 1] = data[:, :width]
        return table

    data_dir = param['pkg_dir'] + 'forward_mod/Data/'
    param['reaction_tables'] = {
        'ReactionR': _load_reaction_table(data_dir + 'Reaction_R.txt', 6),
        'ReactionM': _load_reaction_table(data_dir + 'Reaction_M.txt', 4),
        'ReactionP': _load_reaction_table(data_dir + 'Reaction_P.txt', 8),
        'ReactionT': _load_reaction_table(data_dir + 'Reaction_T.txt', 3),
    }
    return param


def load_photolysis(param):
    species_path = param['pkg_dir'] + 'forward_mod/Data/species_Earth_Full.dat'
    species_data = np.genfromtxt(species_path, dtype=str, skip_header=1)
    if species_data.ndim == 1 and species_data.size:
        species_data = species_data.reshape(1, -1)

    species_by_num = {}
    for row in species_data:
        if row.size >= 3:
            species_by_num[int(row[2])] = row[0]

    reaction_p = np.asarray(param['reaction_tables']['ReactionP'], dtype=int)
    required_species = sorted({int(std_num) for std_num in reaction_p[1:, 1] if int(std_num) > 0})

    photolysis_tables = {}
    for std_num in required_species:
        species_name = species_by_num.get(std_num)
        if species_name is None:
            continue
        file_path = param['pkg_dir'] + 'forward_mod/' + species_name
        if not os.path.exists(file_path):
            continue
        data = np.loadtxt(file_path)
        data = np.atleast_2d(np.asarray(data, dtype=float))
        order = np.argsort(data[:, 0])
        photolysis_tables[std_num] = {
            'name': species_name,
            'data': data[order],
        }

    param['species_data'] = species_data
    param['species_by_num'] = species_by_num
    param['photolysis_tables'] = photolysis_tables
    return param


def load_cia(param):
    def _crop_cia_wavelengths(cia_tables):
        min_wl_nm = (param['min_wl'] - 0.05) * 1e3
        max_wl_nm = (param['max_wl'] + 0.05) * 1e3

        for table in cia_tables.values():
            wl = table['wavelength']
            strt = find_nearest(wl, min_wl_nm) - 20
            end = find_nearest(wl, max_wl_nm) + 20
            strt = max(0, strt)
            end = min(len(wl), end)
            if end <= strt:
                strt = 0
                end = len(wl)
            table['wavelength'] = np.ascontiguousarray(wl[strt:end])
            table['values'] = np.ascontiguousarray(table['values'][strt:end, :])

    def _resample_cia_to_opacity_grid(cia_tables, target_wl_nm):
        target_wl_nm = np.ascontiguousarray(np.asarray(target_wl_nm, dtype=float))
        for table in cia_tables.values():
            source_wl = np.asarray(table['wavelength'], dtype=float)
            source_values = np.asarray(table['values'], dtype=float)
            resampled = np.empty((target_wl_nm.size, source_values.shape[1]), dtype=source_values.dtype)
            for col in range(source_values.shape[1]):
                resampled[:, col] = np.interp(
                    target_wl_nm,
                    source_wl,
                    source_values[:, col],
                    left=0.0,
                    right=0.0,
                )
            table['wavelength'] = target_wl_nm.copy()
            table['values'] = np.ascontiguousarray(resampled)

    cia_dir = param['pkg_dir'] + 'forward_mod/opac/cia/'
    cia_files = {
        'H2H2': 'H2-H2_CIA.dat',
        'H2He': 'H2-He_CIA.dat',
        'H2H': 'H2-H_CIA.dat',
        'N2H2': 'N2-H2_CIA.dat',
        'N2N2': 'N2-N2_CIA.dat',
        'CO2CO2': 'CO2-CO2_CIA.dat',
        'O2O2': 'O2-O2_CIA.dat',
        'O2N2': 'O2-N2_CIA.dat',
    }

    cia_tables = {}
    cia_temp = None
    for label, fname in cia_files.items():
        path = cia_dir + fname
        with open(path, 'r') as fim:
            header = fim.readline().split()
        temp_grid = np.array([float(value) for value in header[1:]], dtype=float)
        data = np.loadtxt(path, skiprows=1)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        cia_tables[label] = {
            'wavelength': data[:, 0].astype(float),
            'values': data[:, 1:].astype(float),
        }
        if cia_temp is None:
            cia_temp = temp_grid

    param['cia'] = {
        'temperature_grid': cia_temp,
        'tables': cia_tables,
    }

    _crop_cia_wavelengths(param['cia']['tables'])

    if not param['fit_T'] and param['PT_profile_type'] == 'isothermal':
        target_T = float(param['Tp'])
        if cia_temp.size == 0:
            raise ValueError("CIA temperature grid is empty.")

        if cia_temp.size == 1 or target_T <= cia_temp[0]:
            temp_idx = 0
            for table in param['cia']['tables'].values():
                table['values'] = np.ascontiguousarray(table['values'][:, temp_idx:temp_idx + 1])
        elif target_T >= cia_temp[-1]:
            temp_idx = cia_temp.size - 1
            for table in param['cia']['tables'].values():
                table['values'] = np.ascontiguousarray(table['values'][:, temp_idx:temp_idx + 1])
        else:
            upper_idx = np.searchsorted(cia_temp, target_T, side='left')
            lower_idx = upper_idx - 1
            lower_T = cia_temp[lower_idx]
            upper_T = cia_temp[upper_idx]

            if target_T == upper_T:
                for table in param['cia']['tables'].values():
                    table['values'] = np.ascontiguousarray(table['values'][:, upper_idx:upper_idx + 1])
            elif target_T == lower_T:
                for table in param['cia']['tables'].values():
                    table['values'] = np.ascontiguousarray(table['values'][:, lower_idx:lower_idx + 1])
            else:
                log_lower_T = np.log(lower_T)
                log_upper_T = np.log(upper_T)
                log_target_T = np.log(target_T)
                for table in param['cia']['tables'].values():
                    lower_values = table['values'][:, lower_idx]
                    upper_values = table['values'][:, upper_idx]
                    weight = lower_values.dtype.type((log_target_T - log_lower_T) / (log_upper_T - log_lower_T))
                    table['values'] = (lower_values + (upper_values - lower_values) * weight)[:, np.newaxis]

        param['cia']['temperature_grid'] = np.array([target_T], dtype=cia_temp.dtype)

    if 'opacw' in param:
        target_wl_nm = np.asarray(param['opacw'][0], dtype=float) * 1e9
        _resample_cia_to_opacity_grid(param['cia']['tables'], target_wl_nm)

    return param


def load_cross(param, for_plotting=False):
    def _readcross(fname, read_grids=True):
        NTEMP = 20  # Number of temperature values (from Row 1)
        NPRESSURE = 10  # Number of pressure values (from Row 2)

        opac = []

        with open(fname, 'r') as fim:
            lines = fim.readlines()

        line_idx = 0  # Start from the first line

        if read_grids:
            wave = []

            # Read the temperature values from the first line
            temp_line = lines[line_idx].strip()
            temp_values = [float(x) for x in temp_line.split()]
            if len(temp_values) != NTEMP:
                raise ValueError(f"Expected {NTEMP} temperature values, got {len(temp_values)}")
            temp = temp_values
            line_idx += 1  # Move to the next line

            # Read the pressure values from the second line
            pres_line = lines[line_idx].strip()
            pres_values = [float(x) for x in pres_line.split()]
            if len(pres_values) != NPRESSURE:
                raise ValueError(f"Expected {NPRESSURE} pressure values, got {len(pres_values)}")
            pres = pres_values
            line_idx += 1  # Move to the next line
        else:
            line_idx = 2

        # Now read the data blocks
        while line_idx < len(lines):
            # Read wave number (should be a line with 1 value)
            wave_line = lines[line_idx].strip()
            if not wave_line:
                line_idx += 1
                continue  # Skip empty lines
            if read_grids:
                wave.append(float(wave_line))
            line_idx += 1

            # Read NPRESSURE blocks of data
            opac_block = []
            for _ in range(NPRESSURE):
                if line_idx >= len(lines):
                    raise ValueError("Unexpected end of file while reading data block")

                # Read initial opacity values
                data_line = lines[line_idx].strip()
                data_values = data_line.split()
                opacities = [float(val) for val in data_values[1:]]
                line_idx += 1

                # If opacities are split across multiple lines
                while len(opacities) < NTEMP:
                    if line_idx >= len(lines):
                        raise ValueError("Unexpected end of file while reading opacities")
                    data_line = lines[line_idx].strip()
                    data_values = data_line.split()
                    opacities.extend([float(val) for val in data_values])
                    line_idx += 1

                if len(opacities) > NTEMP:
                    opacities = opacities[:NTEMP]  # Ensure exact size
                opac_block.append(opacities)

            opac.append(opac_block)

        del lines

        opac = np.array(opac)  # Shape: (number of wave numbers, NPRESSURE, NTEMP)
        opac = opac.transpose(1, 2, 0)

        if not read_grids:
            return opac

        # Convert lists to NumPy arrays for easier handling
        temp = np.array(temp).reshape(1, -1)  # Shape: (1, NTEMP)
        pres = np.array(pres).reshape(1, -1)  # Shape: (1, NPRESSURE)
        wave = np.array(wave).reshape(1, -1)  # Shape: (1, number of wave numbers)

        return temp, pres, wave, opac

    if param['opac_data'] is not None:
        if param['opac_dir'] is not None:
            opac_dir = param['opac_dir'] + param['opac_data'] + '/'
        else:
            opac_dir = param['pkg_dir'] + 'forward_mod/opac/' + param['opac_data'] + '/'

        molecules = param['fit_molecules']
        gas_fill = param['gas_fill']
        if gas_fill is not None:
            molecules = molecules + [gas_fill]

        for idx, mol in enumerate(molecules):
            if idx == 0:
                param['opact'], param['opacp'], param['opacw'], param['opac' + mol.lower()] = _readcross(opac_dir + 'opac' + mol + '.dat')
            else:
                param['opac' + mol.lower()] = _readcross(opac_dir + 'opac' + mol + '.dat', read_grids=False)
            param['opac' + mol.lower()] = np.maximum(param['opac' + mol.lower()], 0.0)
            # making opacity files float32 to save space (cut memory usage by half)
            if param['use_float32']:
                param['opac' + mol.lower()] = np.array(param['opac' + mol.lower()], dtype=np.float32)

    if not for_plotting:

        strt = find_nearest(param['opacw'][0] * 1e6, param['min_wl'] - 0.05) - 20
        end = find_nearest(param['opacw'][0] * 1e6, param['max_wl'] + 0.05) + 20
        param['opacw'] = (param['opacw'][0][strt:end]).reshape(1, -1)

        if not param['fit_T'] and param['PT_profile_type'] == 'isothermal':
            strtT = find_nearest(param['opact'][0], float(param['Tp']))
            endT = strtT
            opac_t = np.asarray(param['opact'][0])
            param['opact'] = np.array([[float(param['Tp'])]], dtype=opac_t.dtype)
        elif param['fit_T'] and param['PT_profile_type'] == 'isothermal':
            strtT = find_nearest(param['opact'][0], 50.0)
            endT = find_nearest(param['opact'][0], 700.0)
            param['opact'] = (param['opact'][0][strtT:endT+1]).reshape(1,-1)
        else:
            strtT = 0
            endT = len(param['opact'][0])

        endP = find_nearest(param['opacp'][0], param['P_standard'][-1])
        param['opacp'] = (param['opacp'][0][:endP+1]).reshape(1,-1)
        
        molecules = param['fit_molecules']
        gas_fill = param['gas_fill']
        if gas_fill is not None:
            molecules = molecules + [gas_fill]
        for mol in molecules:
            param['opac' + mol.lower()] = param['opac' + mol.lower()][:endP + 1, strtT:endT + 1, strt:end]

    _warn_if_stellar_grid_is_too_coarse(param)
    return param


def retrieval_par_and_npar(param):
    parameters = []
    if param['fit_p0']:
        parameters.append("P$_0$")
    if param['fit_wtr_cld'] and param['PT_profile_type'] == 'isothermal':
        parameters.append("Log(P$_{top, H_2O}$)")
        parameters.append("Log(D$_{H_2O}$)")
        parameters.append("Log(CR$_{H_2O}$)")
    if param['fit_amm_cld'] and param['PT_profile_type'] == 'isothermal':
        parameters.append("Log(P$_{top, NH_3}$)")
        parameters.append("Log(D$_{NH_3}$)")
        parameters.append("Log(CR$_{NH_3}$)")
    if param['fit_H2O']:
        parameters.append("H$_2$O")
    if param['fit_NH3']:
        parameters.append("NH$_3$")
    if param['fit_CH4']:
        parameters.append("CH$_4$")
    if param['fit_H2S']:
        parameters.append("H$_2$S")
    if param['fit_SO2']:
        parameters.append("SO$_2$")
    if param['fit_CO2']:
        parameters.append("CO$_2$")
    if param['fit_CO']:
        parameters.append("CO")
    if param['fit_O2']:
        parameters.append("O$_2$")
    if param['fit_O3']:
        parameters.append("O$_3$")
    if param['fit_N2O']:
        parameters.append("N$_2$O")
    if param['fit_N2']:
        parameters.append("N$_2$")
    if param['fit_H2']:
        parameters.append("H$_2$")
    if param['fit_ag']:
        if param['surface_albedo_parameters'] == int(1):
            parameters.append("$a_{surf}$")
        elif param['surface_albedo_parameters'] == int(3):
            parameters.append("$a_{surf, 1}$")
            parameters.append("$a_{surf, 2}$")
            parameters.append(r"$\lambda_{surf, 1}$")
        elif param['surface_albedo_parameters'] == int(5):
            parameters.append("$a_{surf, 1}$")
            parameters.append("$a_{surf, 2}$")
            parameters.append("$a_{surf, 3}$")
            parameters.append(r"$\lambda_{surf, 1}$")
            parameters.append(r"$\lambda_{surf, 2}$")
    if param['fit_T']:
        if param['PT_profile_type'] == 'isothermal':
            parameters.append("T$_p$")
        elif param['PT_profile_type'] == 'parametric':
            parameters.append(r"$\kappa_{th}$")
            parameters.append(r"$\gamma$")
            parameters.append("$\beta$")
            if param['fit_Tint']:
                parameters.append("T$_{int}$")
    if param['fit_cld_frac']:
        parameters.append("Log(cld frac)")
    if param['fit_g']:
        parameters.append("Log(g)")
    if param['fit_Mp']:
        parameters.append("M$_p$")
    if param['fit_Rp']:
        parameters.append("R$_p$")
    if param['fit_p_size']:
        parameters.append("Log(P$_{size}$)")
    if param['fit_phi']:
        if param['obs_numb'] is None:
            parameters.append(r"$\phi$")
        else:
            for obs in range(0, param['obs_numb']):
                parameters.append(r"$\phi_" + str(obs) + "$")

    return parameters, len(parameters)


def write_stats_summary_files(param, prefix, multinest_stats, n_fitted_parameters):
    def _retrieval_n_data_points(param):
        if param['obs_numb'] is None:
            return int(len(param['spectrum']['Fplanet']))

        n_data = 0
        for obs in range(0, int(param['obs_numb'])):
            n_data += len(param['spectrum'][str(obs)]['Fplanet'])
        return int(n_data)

    def _retrieval_loglike_constant(param):
        def _validated_error_array(err):
            err = np.asarray(err, dtype=float)
            if err.size == 0:
                return np.array([], dtype=float)
            if not np.all(np.isfinite(err)) or np.any(err <= 0.0):
                return None
            return err

        norm = np.sqrt(2.0 * math.pi)
        if param['obs_numb'] is None:
            err = _validated_error_array(param['spectrum']['error_p'])
            if err is None:
                return np.nan
            return float(np.sum(np.log(err * norm)))

        logc = 0.0
        for obs in range(0, int(param['obs_numb'])):
            err = _validated_error_array(param['spectrum'][str(obs)]['error_p'])
            if err is None:
                return np.nan
            logc += float(np.sum(np.log(err * norm)))
        return float(logc)

    def _lnl_hat_from_chain_file(file_path):
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
                col2 = np.array([float(data[1])], dtype=float)
            except (TypeError, ValueError):
                return None
        else:
            try:
                col2 = np.asarray(data[:, 1], dtype=float)
            except (TypeError, ValueError):
                return None

        if col2.size == 0:
            return None

        finite = col2[np.isfinite(col2)]
        if finite.size == 0:
            return None

        # MultiNest chain column 2 is -2*ln(L); best fit minimizes this column.
        return float(-0.5 * np.min(finite))

    def _bpics_from_chain_file(file_path, n_params):
        """Calculates the simplified Bayesian Predictive Information Criterion
        (BPICs) described in Ando (2011) for the given sample log-likelihoods
        and number of parameters. Models with a lower BPICS are preferred.

        Citation: Ando (2011), DOI 10.1080/01966324.2011.10737798

        Args:
            logl_samples: The natural log of the likelihood of posterior draws
                from an MCMC run of the model.
            n_params: The number of parameters for the model.
            log_weights: The weights of the samples given, mainly for nested
                sampling posteriors. For equally weighted samples, leave as
                None.

        Returns:
            The computed BPICS as a float.
        """
        # Ando (2011) , DOI 10.1080/01966324.2011.10737798
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
                logl_samples = np.array([-0.5 * float(data[1])], dtype=float)
            except (TypeError, ValueError):
                return None
            weights = None
            if len(data) >= 1:
                try:
                    sample_weight = float(data[0])
                except (TypeError, ValueError):
                    sample_weight = np.nan
                if np.isfinite(sample_weight) and sample_weight > 0.0:
                    weights = np.array([sample_weight], dtype=float)
        else:
            if data.shape[1] < 2:
                return None
            try:
                logl_samples = -0.5 * np.asarray(data[:, 1], dtype=float)
            except (TypeError, ValueError):
                return None
            try:
                weights = np.asarray(data[:, 0], dtype=float)
            except (TypeError, ValueError):
                weights = None

        finite = np.isfinite(logl_samples)
        if weights is not None:
            finite &= np.isfinite(weights) & (weights > 0.0)

        logl_samples = logl_samples[finite]
        if logl_samples.size == 0:
            return None

        if weights is not None:
            weights = weights[finite]
            weight_sum = np.sum(weights)
            if not np.isfinite(weight_sum) or weight_sum <= 0.0:
                weights = None
            else:
                weights = weights / weight_sum

        mean_logl = np.average(logl_samples, weights=weights)
        if not np.isfinite(mean_logl):
            return None

        return float((-2.0 * mean_logl) + (2.0 * float(n_params)))

    def _dic_from_chain_file(file_path):
        """Calculates the Deviance Information Criterion (DIC) for the given
        sample log-likelihoods, using the Ando (2011) variant and the Gelman
        (2014) number of effective parameters formula. Models with lower DIC
        are preferred.

        Args:
            logl_samples: The natural log of the likelihood of posterior draws
                from the MCMC run.

        Returns:
            The computed DIC as a float.
        """
        # Ando (2011); Gelman (2014)
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
                logl_samples = np.array([-0.5 * float(data[1])], dtype=float)
            except (TypeError, ValueError):
                return None
            weights = None
            if len(data) >= 1:
                try:
                    sample_weight = float(data[0])
                except (TypeError, ValueError):
                    sample_weight = np.nan
                if np.isfinite(sample_weight) and sample_weight > 0.0:
                    weights = np.array([sample_weight], dtype=float)
        else:
            if data.shape[1] < 2:
                return None
            try:
                logl_samples = -0.5 * np.asarray(data[:, 1], dtype=float)
            except (TypeError, ValueError):
                return None
            try:
                weights = np.asarray(data[:, 0], dtype=float)
            except (TypeError, ValueError):
                weights = None

        finite = np.isfinite(logl_samples)
        if weights is not None:
            finite &= np.isfinite(weights) & (weights > 0.0)

        logl_samples = logl_samples[finite]
        if logl_samples.size == 0:
            return None

        if weights is not None:
            weights = weights[finite]
            weight_sum = np.sum(weights)
            if not np.isfinite(weight_sum) or weight_sum <= 0.0:
                weights = None
            else:
                weights = weights / weight_sum

        mean_logl = np.average(logl_samples, weights=weights)
        if not np.isfinite(mean_logl):
            return None

        if weights is None:
            p_d = 2.0 * np.var(logl_samples)
        else:
            variance = np.average((logl_samples - mean_logl) ** 2, weights=weights)
            p_d = 2.0 * variance

        dic = (-2.0 * mean_logl) + (3.0 * p_d)
        if not np.isfinite(dic):
            return None

        return float(dic)

    def _waic_from_pointwise_logl_file(file_path, n_expected_points=None):
        """Compute the WAIC from a saved pointwise log-likelihood array.

        Pointwise log-likelihood values are not usually recorded by MCMC codes,
        so they must be preserved explicitly. In Dynesty, this can be done by
        setting ``blob=True`` in the sampler initialization and modifying the
        likelihood function to return both the summed and pointwise values. The
        pointwise log-likelihood can then be retrieved from the results object's
        ``blob`` attribute. EMCEE has a similar mechanism.

        Citation: Watanabe (2010) [no DOI]

        Args:
            file_path: Path to the saved pointwise log-likelihood array.
            n_expected_points: Optional expected number of data points used to
                infer whether the loaded array should be transposed.

        Returns:
            The WAIC statistic for the model, or ``None`` if it cannot be
            computed from the file contents.
        """
        # Watanabe (2010)
        if not os.path.isfile(file_path):
            return None

        try:
            pointwise_logl = np.loadtxt(file_path)
        except (OSError, ValueError):
            return None

        pointwise_logl = np.asarray(pointwise_logl, dtype=float)
        if pointwise_logl.ndim != 2:
            return None
        if pointwise_logl.shape[0] == 0 or pointwise_logl.shape[1] == 0:
            return None

        # ExoReL stores likelihood samples as (n_samples, n_points).
        if n_expected_points is not None:
            if pointwise_logl.shape[1] == int(n_expected_points):
                pass
            elif pointwise_logl.shape[0] == int(n_expected_points):
                pointwise_logl = pointwise_logl.T

        finite_cols = np.all(np.isfinite(pointwise_logl), axis=0)
        pointwise_logl = pointwise_logl[:, finite_cols]
        if pointwise_logl.shape[1] == 0:
            return None

        n_samples = pointwise_logl.shape[0]
        fit_term = sp.special.logsumexp(pointwise_logl, axis=0, b=(1.0 / float(n_samples)))
        penalty_term = np.var(pointwise_logl, axis=0)
        waic = -2.0 * (np.sum(fit_term) - np.sum(penalty_term))
        if not np.isfinite(waic):
            return None

        return float(waic)
    
    def _lnl_hat_per_mode_from_post_separate(post_separate_path):
        if not os.path.isfile(post_separate_path):
            return []

        mode_lnl_hat = []
        current_min_col2 = None
        empty_rows = 0
        with open(post_separate_path, 'r') as f:
            for idx, raw_line in enumerate(f):
                if idx <= 2:
                    continue

                line = raw_line.strip()
                if len(line) == 0:
                    empty_rows += 1
                    continue

                if empty_rows >= 2 and current_min_col2 is not None:
                    mode_lnl_hat.append(float(-0.5 * current_min_col2))
                    current_min_col2 = None
                empty_rows = 0

                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    col2 = float(parts[1])
                except (TypeError, ValueError):
                    continue
                if (current_min_col2 is None) or (col2 < current_min_col2):
                    current_min_col2 = col2

        if current_min_col2 is not None:
            mode_lnl_hat.append(float(-0.5 * current_min_col2))

        return mode_lnl_hat
    
    def _chi_square_from_best_fit_file(param, best_fit_path):
        if param['obs_numb'] is not None or not os.path.isfile(best_fit_path):
            return None

        best_fit = np.loadtxt(best_fit_path)
        if best_fit.ndim != 2 or best_fit.shape[1] < 2:
            return None

        model_wl = best_fit[:, 0]
        model_flux = best_fit[:, 1]

        if param['spectrum']['bins']:
            wl_bins = np.array([param['spectrum']['wl_low'], param['spectrum']['wl_high']]).T
            model_at_data = custom_spectral_binning(wl_bins, model_wl, model_flux, bins=True)
        else:
            model_at_data = spectres(param['spectrum']['wl'], model_wl, model_flux, fill=False)

        chi = (param['spectrum']['Fplanet'] - model_at_data) / param['spectrum']['error_p']
        return float(np.sum(chi ** 2.0))
    
    def _summary_mode_indices(multinest_stats, filter_multi_solutions):
        modes = multinest_stats.get('modes', [])
        if len(modes) == 0:
            return []

        if len(modes) == 1:
            return [0]

        local_loge = []
        for mode in modes:
            value = mode.get('local log-evidence')
            if value is None:
                value = -np.inf
            local_loge.append(float(value))

        max_idx = int(np.argmax(local_loge))
        threshold = 11.0 if filter_multi_solutions else 1000.0

        selected = [max_idx]
        for mode_idx in range(0, len(modes)):
            if mode_idx == max_idx:
                continue
            if (local_loge[max_idx] - local_loge[mode_idx]) < threshold:
                selected.append(mode_idx)

        return selected
    
    def _safe_json_number(value):
        if value is None:
            return None
        value = float(value)
        if np.isfinite(value):
            return value
        return None
    
    def _summary_float_str(value):
        if value is None:
            return 'nan'
        value = float(value)
        if not np.isfinite(value):
            return 'nan'
        return f'{value:.2f}'
    
    modes = multinest_stats.get('modes', [])
    if len(modes) == 0:
        return

    mode_indices = _summary_mode_indices(multinest_stats, param.get('filter_multi_solutions', False))
    if len(mode_indices) == 0:
        return

    n_data_points = _retrieval_n_data_points(param)
    n_fit = int(n_fitted_parameters)
    dof = int(n_data_points - n_fit)
    loglike_const = _retrieval_loglike_constant(param)

    mode_lnl_hat = []
    if len(modes) > 1:
        mode_lnl_hat = _lnl_hat_per_mode_from_post_separate(prefix + 'post_separate.dat')
    else:
        single_mode_lnl_hat = _lnl_hat_from_chain_file(prefix + '.txt')
        if single_mode_lnl_hat is not None:
            mode_lnl_hat = [single_mode_lnl_hat]

    summary_data = {'solutions': {}}
    txt_lines = []

    for mode_pos, mode_idx in enumerate(mode_indices):
        mode = modes[mode_idx]

        lnl_hat = None
        if mode_idx < len(mode_lnl_hat):
            lnl_hat = mode_lnl_hat[mode_idx]
        if lnl_hat is None:
            lnl_hat = _lnl_hat_from_chain_file(prefix + f'solution{mode_idx}.txt')
        if lnl_hat is None:
            lnl_hat = mode.get('maximum log-likelihood')
        if lnl_hat is not None:
            lnl_hat = float(lnl_hat)

        waic = _waic_from_pointwise_logl_file(
            param['out_dir'] + f'loglike_per_datapoint_sol{mode_idx}.dat',
            n_expected_points=n_data_points,
        )
        if waic is None and param.get('filter_multi_solutions', False):
            waic = _waic_from_pointwise_logl_file(
                param['out_dir'] + f'loglike_per_datapoint_sol{mode_pos}.dat',
                n_expected_points=n_data_points,
            )
        if waic is None:
            waic = np.nan

        bpics = _bpics_from_chain_file(prefix + f'solution{mode_idx}.txt', n_fit)
        if bpics is None and len(modes) == 1:
            bpics = _bpics_from_chain_file(prefix + '.txt', n_fit)
        if bpics is None:
            bpics = np.nan

        dic = _dic_from_chain_file(prefix + f'solution{mode_idx}.txt')
        if dic is None and len(modes) == 1:
            dic = _dic_from_chain_file(prefix + '.txt')
        if dic is None:
            dic = np.nan

        chi_square = _chi_square_from_best_fit_file(param, param['out_dir'] + f'Best_fit_sol{mode_idx}.dat')
        if chi_square is None and param.get('filter_multi_solutions', False):
            # In filtered runs, best-fit files can be indexed by filtered order.
            chi_square = _chi_square_from_best_fit_file(param, param['out_dir'] + f'Best_fit_sol{mode_pos}.dat')

        if chi_square is None and lnl_hat is not None and np.isfinite(lnl_hat) and np.isfinite(loglike_const):
            chi_square = float(-2.0 * (lnl_hat + loglike_const))
        if chi_square is None:
            chi_square = np.nan

        if lnl_hat is not None and np.isfinite(lnl_hat):
            aic = float((2.0 * n_fit) - (2.0 * lnl_hat))
            if n_data_points > (n_fit + 1):
                aicc = float(aic + ((2.0 * n_fit * (n_fit + 1)) / (n_data_points - n_fit - 1)))
            else:
                aicc = np.nan
            if n_data_points > 0:
                bic = float((np.log(n_data_points) * n_fit) - (2.0 * lnl_hat))
            else:
                bic = np.nan
        else:
            aic = np.nan
            aicc = np.nan
            bic = np.nan

        if dof != 0 and np.isfinite(chi_square):
            reduced_chi_square = float(chi_square / dof)
        else:
            reduced_chi_square = np.nan

        ln_z = mode.get('local log-evidence', np.nan)
        ln_z_err = mode.get('local log-evidence error', np.nan)

        txt_lines.append(f'*** SOLUTION {mode_idx} ***')
        txt_lines.append('############### SUMMARY STATISTICS ###############')
        txt_lines.append('')
        txt_lines.append(f'chi-square (d.o.f) = {_summary_float_str(chi_square)} ({dof})')
        txt_lines.append(f'Reduced chi-square = {_summary_float_str(reduced_chi_square)}')
        txt_lines.append(f'ln Z               = {_summary_float_str(ln_z)} +- {_summary_float_str(ln_z_err)}')
        txt_lines.append(f'AIC                = {_summary_float_str(aic)}')
        txt_lines.append(f'AICc               = {_summary_float_str(aicc)}')
        txt_lines.append(f'WAIC               = {_summary_float_str(waic)}')
        txt_lines.append(f'BIC                = {_summary_float_str(bic)}')
        txt_lines.append(f'DIC                = {_summary_float_str(dic)}')
        txt_lines.append(f'BPICs              = {_summary_float_str(bpics)}')
        txt_lines.append('')
        txt_lines.append('##################################################')
        txt_lines.append('')

        summary_data['solutions'][f'solution{mode_idx}'] = {
            'solution_index': int(mode_idx),
            'n_data_points': int(n_data_points),
            'n_fitted_parameters': int(n_fit),
            'degrees_of_freedom': int(dof),
            'chi_square': _safe_json_number(chi_square),
            'reduced_chi_square': _safe_json_number(reduced_chi_square),
            'ln_Z': _safe_json_number(ln_z),
            'ln_Z_error': _safe_json_number(ln_z_err),
            'AIC': _safe_json_number(aic),
            'AICc': _safe_json_number(aicc),
            'WAIC': _safe_json_number(waic),
            'BIC': _safe_json_number(bic),
            'DIC': _safe_json_number(dic),
            'BPICs': _safe_json_number(bpics),
            'max_log_likelihood': _safe_json_number(lnl_hat),
        }

    with open(param['out_dir'] + 'stats_summary.txt', 'w') as f_txt:
        f_txt.write('\n'.join(txt_lines).rstrip() + '\n')

    with open(param['out_dir'] + 'stats_summary.json', 'w') as f_json:
        json.dump(summary_data, f_json, indent=2)


def detect_gen_npar(param):
    n_parameters = 0
    parameters = []
    for key, value in param.items():
        if key.endswith('_range'):
            if key == 'pp_range' or key == 'vmr_range':
                pass
            else:
                parameters.append(key)
                n_parameters += 1

    for i in param['fit_molecules']:
        if param['fit_' + i]:
            parameters.append(i)
            n_parameters += 1

    return n_parameters, parameters


def clr_to_vmr(param, centered_log_ratio):
    c_l_r = []
    for mol in param['fit_molecules']:
        c_l_r.append(centered_log_ratio[mol])
    c_l_r = np.array(c_l_r)
    c_l_r = np.append(c_l_r, -np.sum(c_l_r))
    v_m_r = clr_inv(c_l_r)
    for i, mol in enumerate(param['fit_molecules']):
        param['vmr_' + mol] = v_m_r[i]
    param['vmr_' + param['gas_fill']] = v_m_r[-1]

    return param, np.sum(v_m_r)


def elapsed(t):
    milliseconds = round(t / 10 ** 6., 0)  # in milliseconds
    if milliseconds > 10 ** 3:
        seconds = int(milliseconds / 10 ** 3.)  # in seconds
        milliseconds = milliseconds - (seconds * (10 ** 3.))
        if seconds / 60. > 1:
            minutes = int(seconds / 60.)
            seconds = int(seconds - (minutes * 60.))
            if minutes / 60. > 1:
                hours = int(minutes / 60.)
                minutes = int(minutes - (hours * 60.))
                if hours / 24. > 1:
                    days = int(hours / 24.)
                    hours = int(hours - (days * 24.))
                    print('ExoReL runtime : ' + str(days) + ' days, ' + str(hours) + ' hours, ' + str(
                        minutes) + ' minutes, and ' + str(seconds) + ' seconds')
                else:
                    print('ExoReL runtime : ' + str(hours) + ' hours, ' + str(minutes) + ' minutes, and ' + str(
                        seconds) + ' seconds')
            else:
                print('ExoReL runtime : ' + str(minutes) + ' minutes and ' + str(seconds) + ' seconds')
        else:
            print('ExoReL runtime : ' + str(seconds) + ' seconds and ' + str(milliseconds) + ' milliseconds')
    else:
        print('ExoReL runtime : ' + str(milliseconds) + ' milliseconds')


def add_noise(param, data, noise_model=0):
    """
    Calculates one of two types of noise based on the chosen model

    Parameters
    ----------
    param : dict
        dictionary of settings, noise specific settings can already be set or
        left blank and set in this function.
    data : array
        planetary contrast ratio or planet flux. Either flux or contrast must be supplied.
    noise_model : int, optional
        which noise model (0 or 1) should be used. The default is 0.

    Returns
    -------
    err : array
        The errorbar at each wavelength.
    """

    def gaussian_noise(spec, no_less_zero=False):
        '''
        Adds gaussian noise with sigma=err to spectrum

        Parameters
        ----------
        spectrum : array-like
            planet spectrum or contrast.
        err : array-like
            error bars on spectrum for each point.

        Returns
        -------
        spec_with_error : array
            spectrum with gaussian noise added.
        '''

        spec = spec + 0.0
        for i in range(0, len(spec[:, 1])):
            point = np.random.normal(spec[i, 1], spec[i, 2])
            if no_less_zero:
                while point < 0.0:
                    point = np.random.normal(spec[i, 1], spec[i, 2])
            spec[i, 1] = point + 0.0
        return spec

    def chi_square(data, model, deg=None):
        chi = (data[:, 1] - model) / data[:, 2]
        chi = np.sum(chi ** 2.)

        if deg is None:
            return chi, chi / (len(data[:, 0]) - 1)
        else:
            return chi / deg

    # Check if contrast or flux is given and calculate other
    if param['fp_over_fs']:
        contrast = data[:, 1] + 0.0
        if noise_model != 0:
            # Check if star spectrum exists, and if not load it
            try:
                _star_spectrum_entry(param)['y'][0] += 0.0
            except (KeyError, IndexError, TypeError):
                param = take_star_spectrum(param)

            F_s = _star_spectrum_entry(param)['y'] * (((param['Rs'] * const.R_sun.value) / (param['distance'] * const.pc.value)) ** 2.0)
            F_p = data[:, 1] * F_s
    elif not param['fp_over_fs'] and not param['albedo_calc']:
        F_p = data[:, 1] + 0.0
        # Check if star spectrum exists, and if not load it
        try:
            _star_spectrum_entry(param)['y'][0] += 0.0
        except (KeyError, IndexError, TypeError):
            param = take_star_spectrum(param)

        F_s = _star_spectrum_entry(param)['y'] * (((param['Rs'] * const.R_sun.value) / (param['distance'] * const.pc.value)) ** 2.0)
        contrast = data[:, 1] / F_s
    elif param['albedo_calc'] and not param['fp_over_fs']:
        raise TypeError('Cannot calculate the error on the albedo. Please, provide the contrast ratio or the planetary flux.')

    try:
        param['l0'] += 0.0
    except KeyError:
        param['l0'] = 0.75

    i0 = find_nearest(param['spectrum']['wl'], param['l0'])
    C0 = contrast[i0]
    l0 = param['spectrum']['wl'][i0]

    if noise_model == 0:
        # Check if variables exist, and if not compute or set to defaults
        try:
            param['alpha1'] += 0.0
        except KeyError:
            param['alpha1'] = C0 / param['snr']

        # calculate err
        err = np.ones(len(param['spectrum']['wl'])) * param['alpha1']

    elif noise_model == 1:
        # Check if variables exist, and if not set to defaults
        try:
            param['alpha1'] += 0.0
        except KeyError:
            param['alpha1'] = 0.5 * C0 / (l0 ** 2)
        try:
            param['alpha2'] += 0.0
        except KeyError:
            param['alpha2'] = 1e-11

        # calculate SNR
        R = param['spectrum']['wl'][:-1] / np.diff(param['spectrum']['wl'])
        R = np.append(R, R[-1])
        wl = param['spectrum']['wl'] / l0
        SNR = F_p * wl * (R * (F_p + param['alpha1'] * F_s * (wl ** 2) + param['alpha2'] * F_s)) ** (-0.5)

        try:
            param['alpha3'] += 0.0
        except KeyError:
            param['alpha3'] = param['snr'] / SNR[i0]
        SNR *= param['alpha3']

        # convert SNR to error
        err = contrast / SNR

        try:
            SNR = np.array([param['spectrum']['wl'], SNR]).T
            if param['spectrum']['bins']:
                SNR = np.concatenate((np.array([param['spectrum']['wl_high']]).T, SNR), axis=1)
                SNR = np.concatenate((np.array([param['spectrum']['wl_low']]).T, SNR), axis=1)
            if param['save_snr_array']:
                np.savetxt(param['out_dir'] + 'snr_vs_wavelength.dat', SNR)
        except KeyError:
            pass

    spectrum = np.array([data[:, 0], data[:, 1], err]).T

    if param['gaussian_noise']:
        spec_copy = spectrum + 0.0
        chi = 1.0

        for _ in range(1000):
            new_spec = gaussian_noise(spec_copy, no_less_zero=True)
            if chi_square(new_spec, data[:, 1])[1] < chi:
                chi = chi_square(new_spec, data[:, 1])[1]
                spectrum = new_spec + 0.0
            else:
                pass

    return spectrum


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


def clean_c_files(directory):
    file_list = glob.glob(directory + 'forward_mod/core_*.c')
    if len(file_list) > 0:
        for i in file_list:
            os.system('rm ' + i)

    file_list = glob.glob(directory + 'forward_mod/par_*.h')
    if len(file_list) > 0:
        for i in file_list:
            os.system('rm ' + i)

    file_list = glob.glob(directory + 'forward_mod/None*')
    if len(file_list) > 0:
        for i in file_list:
            os.system('rm ' + i)

    for j in range(0, 10):
        file_list = glob.glob(directory + 'forward_mod/' + str(j) + '*')
        if len(file_list) > 0:
            for i in file_list:
                os.system('rm ' + i)

    file_list = glob.glob(directory + 'forward_mod/Result/Retrieval_*')
    if len(file_list) > 0:
        for i in file_list:
            os.system('rm -rf ' + i)


def reso_range(start, finish, res, bins=False):
    wl_low = [start]
    res = 1. / res
    wl_high = [start + (start * res)]
    while wl_high[-1] < finish:
        wl_low.append(wl_high[-1])
        wl_high.append(wl_low[-1] + (wl_low[-1] * res))

    bns = np.array([wl_low, wl_high]).T

    if not bins:
        return np.mean(bns, axis=1)
    else:
        return bns
