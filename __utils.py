from .__basics import *
import shutil


def default_parameters():
    param = {}

    param['wkg_dir'] = os.getcwd()

    #### [STAR] ####
    param['Rs'] = None  # Star radius
    param['Ts'] = None  # Star temperature
    param['meta'] = None  # star metallicity [M / H]
    param['Loggs'] = None  # Star log surface gravity
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
    param['Rp_prior_type'] = 'independent'  # type of prior function for the planetary radius. Possibilities: independent, M_R_prior, R_M_prior, random_error
    param['Mp_prior_type'] = 'independent'  # type of prior function for the planetary mass. Possibilities: independent, M_R_prior, R_M_prior, random_error

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
    param['plot_models'] = False  # whether to plot spectrum, surface, and atmospheric chemistry graphs
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


def read_parfile(param, parfile=None, json_format=False):
    cwd = os.getcwd()
    if parfile is None:
            print('No parameter file provided. A standard parameter file will be used.')
            pass
    else:
        if json_format:
            with open(parfile, 'r') as f:
                paramdata = json.load(f)
            for key, value in paramdata.items():
                param[key] = value
            del paramdata
        else:
            #print('Reading parfile: "' + parfile + '"')
            with open(cwd + '/' + parfile, 'r') as file:
                paramfile = file.readlines()
            for i in paramfile:
                if i[0] == '%' or i[0] == '\n':
                    pass
                else:
                    paramline = list(i.split('\t'))
                    paramline[-1] = paramline[-1][:-1]
                    if len(paramline) >= 2:
                        try:
                            param[paramline[0]] = float(paramline[-1])
                        except ValueError:
                            if str(paramline[1]) == str(True):
                                param[paramline[0]] = bool(paramline[1])
                            elif str(paramline[1]) == str(False):
                                param[paramline[0]] = bool("")
                            elif str(paramline[1]) == str(None):
                                param[paramline[0]] = None
                            else:
                                param[paramline[0]] = str(paramline[1])

                        if paramline[0] == 'file_output_name':
                            try:
                                param[paramline[0]] = str(int(paramline[1]))
                            except ValueError:
                                param[paramline[0]] = str(paramline[1])
                    else:
                        paramline = str(paramline[0]).split()
                        if paramline[0] == 'mol':
                            param[paramline[0]] = paramline[1].split(',')
                        elif paramline[0] == 'mol_vmr' or paramline[0] == 'range_mol':
                            param[paramline[0]] = paramline[1].split(',')
                            for ob in range(0, len(param[paramline[0]])):
                                param[paramline[0]][ob] = float(param[paramline[0]][ob])
                            if paramline[0] == 'mol_vmr':
                                for num, mol in enumerate(param['mol']):
                                    param['vmr_' + mol] = param['mol_vmr'][num]
                            else:
                                pass
                        else:
                            try:
                                param[paramline[0]] = float(paramline[-1])
                            except ValueError:
                                if str(paramline[1]) == str(True):
                                    param[paramline[0]] = bool(paramline[1])
                                elif str(paramline[1]) == str(False):
                                    param[paramline[0]] = bool("")
                                elif str(paramline[1]) == str(None):
                                    param[paramline[0]] = None
                                else:
                                    param[paramline[0]] = str(paramline[1])

    param['wkg_dir'] = cwd + '/'
    if param['output_directory'] is not None:
        param['out_dir'] = param['wkg_dir'] + param['output_directory']
        if not os.path.isdir(param['out_dir']):
            os.mkdir(param['out_dir'])
        del param['output_directory']
    else:
        param['out_dir'] = param['wkg_dir']

    src = os.path.join(cwd, parfile)
    dst = os.path.join(param['out_dir'], os.path.basename(parfile))
    if not os.path.exists(dst):
        shutil.copy2(src, dst)

    return param


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

    if "Mp_range" in param.keys():
        param['Mp'] = param["Mp_range"][0] + 0.0

    if param['Mp'] <= 0.06:
        param['rocky'] = True
    else:
        param['rocky'] = False
        param['fit_p0'] = False
        param['P0'] = 10 ** 11.5
        param['fit_ag'] = False
        param['Ag'] = 0.0
        param['gas_par_space'] = 'vmr'
        param['gas_fill'] = 'H2'

    if param['Mp'] is not None:
        param['Mp_orig'] = param['Mp'] + 0.0

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

    return param


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
                param['spectrum'] = {}
                if len(spectrum[0, :]) == 3:
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
                    param['spectrum'][str(obs)] = {}
                    if len(spectrum[0, :]) == 3:
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
            mix = np.empty((len(P)))
            # assuming water vmr is limited by saturation pressure:
            mix[-1] = np.nanmin([psat[-1]/P[-1], param['vmr_' + condensed_gas]])
            for i in range(len(P)-2, -1, -1):
                mix[i] = np.nanmin([psat[i]/P[i], mix[i+1]])
        else:
            # if param['Pw_top'] > param['P'][-1]:
            if param['P' + initial_letter + '_top'] > param['P'][-1] or (param['P' + initial_letter + '_top'] + param['cld' + initial_letter + '_depth']) > param['P'][-1]:
                no_cloud = True
            else:
                no_cloud = False

            if not no_cloud:
                pos_cld = int(find_nearest(param['P_standard'], param['P' + initial_letter + '_top']))

                if (param['cld' + initial_letter + '_depth'] + param['P_standard'][pos_cld]) > param['P_standard'][-1]:
                    param['cld' + initial_letter + '_depth'] = param['P_standard'][-1] - param['P_standard'][pos_cld]

                pbot = int(find_nearest(param['P_standard'], (param['cld' + initial_letter + '_depth'] + param['P_standard'][pos_cld])))

                depth_layers = pbot - pos_cld
                if depth_layers == 0:
                    return np.ones((len(param['P']))) * param['vmr_' + condensed_gas]
                else:
                    pass

                mix = np.ones((len(param['P_standard']))) * (param['CR_' + condensed_gas] * param['vmr_' + condensed_gas])
                d = (np.log10(param['vmr_' + condensed_gas]) - np.log10(param['CR_' + condensed_gas] * param['vmr_' + condensed_gas])) / depth_layers
                for i in range(0, len(mix)):
                    if i <= pos_cld:
                        pass
                    elif pos_cld < i <= pos_cld + depth_layers:
                        mix[i] = 10. ** (np.log10(mix[i - 1]) + d)
                    elif i > pos_cld + depth_layers:
                        mix[i] = mix[i - 1]
                mix = mix[:len(param['P'])]
            else:
                mix = np.ones((len(param['P']))) * param['vmr_' + condensed_gas]
    else:
        mix = np.ones((len(param['P']))) * param['vmr_' + condensed_gas]

    return mix


def adjust_VMR(param, all_gases=True, condensed_gas='H2O'):
    if all_gases:
        if param['gas_fill'] is None:
            n_gases = len(param['fit_molecules']) - 1
            mol_to_determine = param['fit_molecules'][1:]
        else:
            n_gases = len(param['fit_molecules'])
            mol_to_determine = param['fit_molecules'][1:]
            mol_to_determine.append(param['gas_fill'])

        ratios = []
        for mol in mol_to_determine[1:]:
            ratios.append(param['vmr_' + mol_to_determine[0]] / param['vmr_' + mol])

        matrx = np.ones((n_gases, n_gases))
        for i in range(0, len(matrx[:-1, 0])):
            for j in range(1, len(matrx[0, :])):
                if i + 1 == j:
                    matrx[i, j] = -ratios[i]
                else:
                    matrx[i, j] = 0.0

        res = np.zeros(n_gases)
        for mol in mol_to_determine:
            param['vmr_' + mol] = np.zeros(len(param['vmr_' + condensed_gas]))

        for i in range(0, len(param['vmr_' + condensed_gas])):
            res[-1] = 1.0 - param['vmr_' + condensed_gas][i]
            v_m_r = np.linalg.solve(matrx, res)
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
            param['ag_x1_range'] = [0.4, 1.0]  # wavelength cut-off albedo
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
                param['Mp_range'] = [param['Mp_orig'] - (5.0 * param['Mp_err']), param['Mp_orig'] + (5.0 * param['Mp_err'])]
                if param['Mp_orig'] - (5.0 * param['Mp_err']) < 0.000032:
                    param['Mp_range'][0] = 0.000032
                if param['Mp_orig'] + (5.0 * param['Mp_err']) > 0.06:
                    param['Mp_range'][1] = 0.06
                param['Rp_range'] = [0.044607088905052314, 0.8921417781010462]  # Planet radius - 0.5 to 10 Earth radii
            elif param['Rp_prior_type'] == 'R_M_prior' and param['Mp_prior_type'] != 'M_R_prior':
                param['Rp_range'] = [0.05174422312986068, 0.19627119118223021]          # 0.58 to 2.2 Earth radii
            elif param['Mp_prior_type'] == 'M_R_prior' and param['Rp_prior_type'] != 'R_M_prior':
                param['Mp_range'] = [0.000032, 0.06292703731012286]                      # 0.01 to 20 Earth masses
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


def model_finalizzation(param, alb_wl, alb, planet_albedo=False, fp_over_fs=False, n_obs=None):
    if not param['wl_native']:
        if param['obs_numb'] is not None:
            wl = param['spectrum'][str(n_obs)]['wl'] + 0.0
        else:
            wl = param['spectrum']['wl'] + 0.0

        if param['spectrum']['bins']:
            wl_bins = np.append(np.array([param['spectrum']['wl_low']]).T, np.array([param['spectrum']['wl_high']]).T, axis=1)
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
        planet_flux = contrast * param['starfx']['y'] * (((param['Rs'] * const.R_sun.value) / (param['distance'] * const.pc.value)) ** 2.0)
        return wl, planet_flux


def take_star_spectrum(param, plot=False):
    directory = param['pkg_dir'] + 'PHO_STELLAR_MODEL/'
    t_star = [int(i) for i in str(param['Ts']) if i.isdigit()]
    if t_star[2] >= 5:
        if t_star[1] == 9:
            t_star[1] = 0
            t_star[0] += 1
        else:
            t_star[1] += 1
    try:
        param['Loggs'] += 0.0
    except (KeyError, TypeError):
        param['Loggs'] = round(np.log(274.20011166 * param['Ms'] / (param['Rs'] ** 2.)), 1)

    logg = [int(i) for i in str(param['Loggs']) if i.isdigit()]
    if 0 <= logg[1] <= 2:
        logg[1] = 0
    elif 2 < logg[1] <= 7:
        logg[1] = 5
    elif 7 < logg[1] <= 9:
        logg[1] = 0
        logg[0] += 1

    if logg[0] <= 1:
        logg[0] = 2
        logg[1] = 5

    if logg[0] > 5:
        logg[0] = 5
        logg[1] = 0

    star_file = 'lte0' + str(t_star[0]) + str(t_star[1]) + '.0-' + str(logg[0]) + '.' + str(logg[1]) + '-0.0a+0.0.BT-Settl.spec.7'

    # print('Loading star spectrum')
    # print('Loading ' + star_file + ' from PHOENIX litterature')

    wl = []
    sp = []

    if param['obs_numb'] is None:
        min_wl, max_wl = (param['spectrum']['wl'][0] - 0.2), (param['spectrum']['wl'][-1] + 0.2)
    else:
        min_wl, max_wl = param['min_wl'], param['max_wl']

    with open(directory + star_file) as fp:
        line = fp.readline()[:25]
        line = line.replace("D", "e")
        while line:
            # line = line[:21] + 'e' + line[22:]
            if min_wl <= float(float(line[:13]) * 1.0e-4) <= max_wl:
                wl.append(float(line[:13]))                                             # Angstrom
                sp.append(float(10.0 ** (float(line[13:]) - 8.0 - 3.0)))                # W/m^2/A
            elif float(float(line[:13]) * 1.0e-4) < min_wl:
                pass
            elif float(float(line[:13]) * 1.0e-4) > max_wl:
                break
            line = fp.readline()[:25]
            line = line.replace("D", "e")

    wl = np.array(wl)
    sp = np.array(sp)

    wl2 = np.linspace(wl[0], wl[-1], num=(len(wl) * 2), endpoint=True)  # Double data pitch

    tck = interp1d(wl, sp)
    sp = tck(wl2)
    wl = np.array(wl2) + 0.0

    new_wl = []
    new_sp = []

    for i in range(0, len(wl) - 1):
        new_wl.append(float((wl[i] + wl[i + 1]) / 2.0) * 1.0e-4)
        new_sp.append(float(trapezoid(np.array([sp[i], sp[i + 1]]), x=np.array([wl[i], wl[i + 1]]))))

    wl = np.array(new_wl)                                                               # micron
    sp = np.array(new_sp)                                                               # W/m^2

    if param['obs_numb'] is not None:
        sp = spectres(param['spectrum']['0']['wl'], wl, sp, fill=False)
    else:
        sp = spectres(param['spectrum']['wl'], wl, sp, fill=False)

    param['starfx'] = {'x': wl, 'y': sp}

    if plot:
        plt.plot(wl, sp, 'k-')
        plt.grid()
        plt.xlim([0.4, 1.0])
        plt.title('Stellar spectrum, R=' + str(int(param['Resolution'])))
        plt.xlabel('Wavelength ($\mu$m)')
        plt.ylabel('Stellar flux (W/m$^2$)')
        plt.savefig(param['wkg_dir'] + 'Retrieval/Star_spectrum.pdf')
        plt.close()

    return param


def pre_load_variables(param):
    if not param['rocky']:
    # Mass-Radius diagram
        if param['fit_Mp'] and param['fit_Rp']:
            if param['Rp_prior_type'] == 'R_M_prior':
                M_R_Fe = np.loadtxt(param['pkg_dir'] + 'forward_mod/Data/Fe_mass_radius_jup.dat')
                M_R_H2O = np.loadtxt(param['pkg_dir'] + 'forward_mod/Data/H2O_mass_radius_jup.dat')
                param['R-M_Fe'] = interp1d(M_R_Fe[:, 1], M_R_Fe[:, 0])
                param['R-M_H2O'] = interp1d(M_R_H2O[:, 1], M_R_H2O[:, 0])
            elif param['Mp_prior_type'] == 'M_R_prior':
                M_R_Fe = np.loadtxt(param['pkg_dir'] + 'forward_mod/Data/Fe_mass_radius_jup.dat')
                M_R_H2O = np.loadtxt(param['pkg_dir'] + 'forward_mod/Data/H2O_mass_radius_jup.dat')
                param['M-R_Fe'] = interp1d(M_R_Fe[:, 0], M_R_Fe[:, 1])
                param['M-R_H2O'] = interp1d(M_R_H2O[:, 0], M_R_H2O[:, 1])
            else:
                pass
    #  Load Mie Calculation Results
    if param['fit_wtr_cld']:
        if param['wtr_cld_type'] == 'liquid':
            data = np.loadtxt(param['pkg_dir'] + 'forward_mod/CrossPlnt/Cross_water_wavelength_250916.dat')
            param['H2OL_r'] = data[:, 0]  # zero-order radius, in micron
            param['H2OL_c'] = data[:, 1:]  # cross-section per droplet, in cm2
            data = np.loadtxt(param['pkg_dir'] + 'forward_mod/CrossPlnt/Albedo_water_wavelength_250916.dat')
            param['H2OL_a'] = data[:, 1:]
            data = np.loadtxt(param['pkg_dir'] + 'forward_mod/CrossPlnt/Geo_water_wavelength_250916.dat')
            param['H2OL_g'] = data[:, 1:]
        elif param['wtr_cld_type'] == 'ice':
            data = np.loadtxt(param['pkg_dir'] + 'forward_mod/CrossPlnt/Cross_ice_wavelength_250916.dat')
            param['H2OL_r'] = data[:, 0]  # zero-order radius, in micron
            param['H2OL_c'] = data[:, 1:]  # cross-section per droplet, in cm2
            data = np.loadtxt(param['pkg_dir'] + 'forward_mod/CrossPlnt/Albedo_ice_wavelength_250916.dat')
            param['H2OL_a'] = data[:, 1:]
            data = np.loadtxt(param['pkg_dir'] + 'forward_mod/CrossPlnt/Geo_ice_wavelength_250916.dat')
            param['H2OL_g'] = data[:, 1:]
        elif param['wtr_cld_type'] == 'mixed' and param['PT_profile_type'] == 'parametric':
            data = np.loadtxt(param['pkg_dir'] + 'forward_mod/CrossPlnt/Cross_water_wavelength_250916.dat')
            param['H2OL_r'] = data[:, 0]  # zero-order radius, in micron
            param['H2OL_c_liquid'] = data[:, 1:]  # cross-section per droplet, in cm2
            data = np.loadtxt(param['pkg_dir'] + 'forward_mod/CrossPlnt/Cross_ice_wavelength_250916.dat')
            param['H2OL_c_ice'] = data[:, 1:]  # cross-section per droplet, in cm2

            data = np.loadtxt(param['pkg_dir'] + 'forward_mod/CrossPlnt/Albedo_water_wavelength_250916.dat')
            param['H2OL_a_liquid'] = data[:, 1:]
            data = np.loadtxt(param['pkg_dir'] + 'forward_mod/CrossPlnt/Albedo_ice_wavelength_250916.dat')
            param['H2OL_a_ice'] = data[:, 1:]

            data = np.loadtxt(param['pkg_dir'] + 'forward_mod/CrossPlnt/Geo_water_wavelength_250916.dat')
            param['H2OL_g_liquid'] = data[:, 1:]
            data = np.loadtxt(param['pkg_dir'] + 'forward_mod/CrossPlnt/Geo_ice_wavelength_250916.dat')
            param['H2OL_g_ice'] = data[:, 1:]
    
    if param['fit_amm_cld']:
        data = np.loadtxt(param['pkg_dir'] + 'forward_mod/CrossPlnt/Cross_ammonia_wavelength_250916.dat')
        param['NH3_r'] = data[:, 0]  # zero-order radius, in micron
        param['NH3_c'] = data[:, 1:]  # cross-section per droplet, in cm2
        data = np.loadtxt(param['pkg_dir'] + 'forward_mod/CrossPlnt/Albedo_ammonia_wavelength_250916.dat')
        param['NH3_a'] = data[:, 1:]
        data = np.loadtxt(param['pkg_dir'] + 'forward_mod/CrossPlnt/Geo_ammonia_wavelength_250916.dat')
        param['NH3_g'] = data[:, 1:]

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
            parameters.append("$\lambda_{surf, 1}$")
        elif param['surface_albedo_parameters'] == int(5):
            parameters.append("$a_{surf, 1}$")
            parameters.append("$a_{surf, 2}$")
            parameters.append("$a_{surf, 3}$")
            parameters.append("$\lambda_{surf, 1}$")
            parameters.append("$\lambda_{surf, 2}$")
    if param['fit_T']:
        if param['PT_profile_type'] == 'isothermal':
            parameters.append("T$_p$")
        elif param['PT_profile_type'] == 'parametric':
            parameters.append("$\kappa_{th}$")
            parameters.append("$\gamma$")
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
            parameters.append("$\phi$")
        else:
            for obs in range(0, param['obs_numb']):
                parameters.append("$\phi_" + str(obs) + "$")

    return parameters, len(parameters)


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
                param['starfx']['y'][0] += 0.0
            except KeyError:
                param = take_star_spectrum(param)

            F_s = param['starfx']['y'] * (((param['Rs'] * const.R_sun.value) / (param['distance'] * const.pc.value)) ** 2.0)
            F_p = data[:, 1] * F_s
    elif not param['fp_over_fs'] and not param['albedo_calc']:
        F_p = data[:, 1] + 0.0
        # Check if star spectrum exists, and if not load it
        try:
            param['starfx']['y'][0] += 0.0
        except KeyError:
            param = take_star_spectrum(param)

        F_s = param['starfx']['y'] * (((param['Rs'] * const.R_sun.value) / (param['distance'] * const.pc.value)) ** 2.0)
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


def Mp_prior(param, cube, rp_value=None):
    """
    Prior function for planetary mass

    Parameters
    ----------
    param : dict
        dictionary of settings.
    cube : float
        Mass value to be converted
    rp_value : float, optional
        Radius value to be used in the Radius-Mass diagram

    Returns
    -------
    Mp_value : float
        mass value to be evaluated according to the appropriate mass prior.
    """

    if rp_value is None:
        if param['Mp_err'] is None:
            Mp_value = (cube * (param['Mp_range'][1] - param['Mp_range'][0])) + param['Mp_range'][0]  # ignorant prior
        elif param['Mp_err'] is not None and param['Mp_prior_type'] == 'gaussian':
            Mp_range = np.linspace(param['Mp_range'][0], param['Mp_range'][1], num=10000, endpoint=True)
            Mp_cdf = sp.stats.norm.cdf(Mp_range, param['Mp_orig'], param['Mp_err'])
            Mp_cdf = np.array([0.0] + list(Mp_cdf) + [1.0])
            Mp_range = np.array([Mp_range[0]] + list(Mp_range) + [Mp_range[-1]])
            Mp_pri = interp1d(Mp_cdf, Mp_range)
            Mp_value = Mp_pri(cube)
    else:
        Mp_value = (cube * (param['R-M_Fe'](rp_value) - param['R-M_H2O'](rp_value))) + param['R-M_H2O'](rp_value)

    return Mp_value


def Rp_prior(param, cube, mp_value=None):
    """
    Prior function for radius

    Parameters
    ----------
    param : dict
        dictionary of settings.
    cube : array
        values used to select Rp.
    mp_value : float, optional
        Mass value to be used in the Mass-Radius diagram

    Returns
    -------
    cube : array
        values of Rp.
    """

    if mp_value is None:
        if param['Rp_err'] is None:
            Rp_value = (cube * (param['Rp_range'][1] - param['Rp_range'][0])) + param['Rp_range'][0]  # ignorant prior
        elif param['Rp_err'] is not None:
            Rp_range = np.linspace(param['Rp_range'][0], param['Rp_range'][1], num=1000)
            Rp_cdf = sp.stats.norm.cdf(Rp_range, param['Rp'], param['Rp_err'])
            Rp_pri = interp1d(Rp_cdf, Rp_range)
            Rp_value = Rp_pri(cube)
    else:
        Rp_value = (cube * (param['M-R_H2O'](mp_value) - param['M-R_Fe'](mp_value))) + param['M-R_Fe'](mp_value)

    return Rp_value


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
