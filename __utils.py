from .__basics import *


def take_standard_parameters(pkg_dir):
    parfile = 'standard_parameters.dat'
    with open(pkg_dir + parfile, 'r') as file:
        paramfile = file.readlines()
    param = {}
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

    param['wkg_dir'] = os.getcwd()
    param['supported_molecules'] = ['H2O', 'NH3', 'CH4', 'H2S', 'SO2', 'CO2', 'CO', 'O2', 'O3', 'N2O', 'N2', 'H2']

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


def default_parameters():
    param = {}

    #### [STAR] ####
    param['Rs'] = None  # Star radius
    param['Ts'] = None  # Star temperature
    param['distance'] = None  # star distance from the Sun [pc]
    param['meta'] = None  # star metallicity [M / H]

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
    param['Tirr'] = 394.109  # Irradiation Temperature at 1 AU related to the Sun case [K]
    param['Tint'] = 110.0  # Intrinsic (internal) Temperature [K]
    param['phi'] = None  # Phase angle [deg]
    param['P0'] = None  # Surface pressure [Pa]
    param['Ag'] = None  # Surface albedo

    return param


def read_parfile(param, parfile=None):
    cwd = os.getcwd()
    if parfile is None:
        print('No parameter file provided. A standard parameter file will be used.')
        pass
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

    if param['obs_numb'] is not None:
        param['obs_numb'] = int(param['obs_numb'])
    param['wkg_dir'] = cwd + '/'
    param['out_dir'] = param['wkg_dir'] + param['output_directory']
    try:
        os.mkdir(param['out_dir'])
    except OSError:
        pass
    del param['output_directory']

    param['contribution'] = False
    param['mol_contr'] = None

    if param['gen_dataset_mode']:
        param['rocky'] = True
        return param
    else:
        pass

    os.system('cp ' + cwd + '/' + parfile + ' ' + param['out_dir'])

    if param['albedo_calc']:
        param['fp_over_fs'] = False

    if param['Mp'] is None or param['Mp'] <= 0.06:
        param['rocky'] = True
    else:
        param['rocky'] = False

    if param['Mp'] is not None:
        param['Mp_orig'] = param['Mp'] + 0.0

    if param['gas_fill'] == 'N2':
        param['fit_N2'] = False

    if param['rocky'] and not param['fit_p0'] and param['P0'] is None and param['gas_par_space'] != 'partial_pressure':
        raise ValueError("Surface pressure (P0) needs to be specified since it is not a free parameter.")

    if param['gas_par_space'] == 'partial_pressure' and param['fit_p0']:
        param['fit_p0'] = False
        print('The parameter "fit_p0" has been set to False since the atmospheric chemistry will be fit in the "partial pressure" parameter space.')

    if param['rocky'] and not param['fit_ag'] and param['Ag'] is None:
        raise ValueError("Surface albedo (Ag) needs to be specified since it is not a free parameter.")
    if param['rocky'] and param['fit_ag'] and param['surface_albedo_parameters'] is None:
        param['surface_albedo_parameters'] = int(1)
        print('Surface albedo parameters number not defined. The parameter "surface_albedo_parameters" has been set to 1.')

    if not param['fit_g'] and not param['fit_Mp'] and not param['fit_Rp']:
        if param['Rp'] is not None:
            param['Rp_provided'] = True
        else:
            param['Rp_provided'] = False

        if param['Mp'] is not None:
            param['Mp_provided'] = True
        else:
            param['Mp_provided'] = False

        if param['gp'] is not None:
            param['gp_provided'] = True
        else:
            param['gp_provided'] = False

        if param['Rp'] is None and param['gp'] is None:
            raise ValueError("If radius, mass, and gravity of the planet are not free parameters, please provide at least a combination of two in the parameter file.")

    if param['cld_frac'] > 1.0 or param['cld_frac'] < 0.0:
        raise ValueError("The cloud fraction should be defined between [0.0, 1.0]. Please check the 'cld_frac' value in the parameter file.")

    if param['optimizer'] == 'multinest':
        param['nlive_p'] = int(param['nlive_p'])
        param['max_modes'] = int(param['max_modes'])
    elif param['optimizer'] == 'dynesty':
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
        try:
            param['Tp'] += 0.0
        except KeyError:
            t1 = ((param['Rs'] * const.R_sun.value) / (2. * param['major-a'] * const.au.value)) ** 0.5
            param['Tp'] = t1 * ((1 - 0.3) ** 0.25) * param['Ts']

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

    if param['rocky']:
        param['P_standard'] = 10. ** np.arange(0.0, 12.01, step=0.01)
    else:
        param['P'] = np.loadtxt(param['pkg_dir'] + 'forward_gas_mod/Data/Pressure_grid.dat')  # in Pa
        param['wavelength_planet'] = np.loadtxt(param['pkg_dir'] + 'forward_gas_mod/Data/wave.dat')  # in nanometer

    if param['obs_numb'] is None:
        if not param['fit_phi']:
            param['phi'] = math.pi * param['phi'] / 180.0
    else:
        for obs in range(0, param['obs_numb']):
            if not param['fit_phi']:
                param['phi' + str(obs)] = math.pi * param['phi' + str(obs)] / 180.0

    return param


def calc_mean_mol_mass(param):
    if not param['rocky']:
        vmr_M = 0.0
        for mol in param['fit_molecules']:
            if mol != 'H2O' and mol != 'NH3':
                vmr_M += param['vmr_' + mol]

        if param['fit_wtr_cld']:
            vmr_M += np.mean(param['watermix'])
        else:
            vmr_M += param['vmr_H2O']

        if param['fit_amm_cld']:
            vmr_M += np.mean(param['ammoniamix'])
        else:
            vmr_M += param['vmr_NH3']

        param['vmr_H2'] = 0.75 * ((10. ** 0.0) - vmr_M)
        param['vmr_He'] = 0.25 * ((10. ** 0.0) - vmr_M)

        param['mean_mol_weight'] = (param['vmr_H2'] * param['mm']['H2']) + \
                                   (param['vmr_He'] * param['mm']['He']) + \
                                   (param['vmr_CH4'] * param['mm']['CH4']) + \
                                   (param['vmr_H2S'] * param['mm']['H2S'])

        if param['fit_wtr_cld']:
            param['mean_mol_weight'] += np.mean(param['watermix']) * param['mm']['H2O']
        else:
            param['mean_mol_weight'] += param['vmr_H2O'] * param['mm']['H2O']

        if param['fit_amm_cld']:
            param['mean_mol_weight'] += np.mean(param['ammoniamix']) * param['mm']['NH3']
        else:
            param['mean_mol_weight'] += param['vmr_NH3'] * param['mm']['NH3']

        if not param['ret_mode']:
            print('VMR H2 \t\t = \t' + str(param['vmr_H2']))
            print('mu \t\t = \t' + str(param['mean_mol_weight']))

    else:
        param['mean_mol_weight'] = np.zeros(len(param['P']))
        for i in range(0, len(param['P'])):
            for mol in param['fit_molecules']:
                param['mean_mol_weight'][i] += param['vmr_' + mol][i] * param['mm'][mol]
            if param['gas_fill'] is not None:
                param['mean_mol_weight'][i] += param['vmr_' + param['gas_fill']][i] * param['mm'][param['gas_fill']]

        if not param['ret_mode']:
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
            spectrum = np.loadtxt(param['pkg_dir'] + 'forward_gas_mod/Data/wl_bins/' + param['wave_file'] + '.dat')
        except KeyError:
            if param['rocky']:
                # standard wavelength bin at R = 500 in 0.15 - 2.0 micron
                spectrum = np.loadtxt(param['pkg_dir'] + 'forward_gas_mod/Data/wl_bins/bins_02_20_R500.dat')
            else:
                # standard wavelength bin at R = 500 in the optical wavelength 0.4 - 1.0 micron
                spectrum = np.loadtxt(param['pkg_dir'] + 'forward_gas_mod/Data/wl_bins/bins_04_10_R500.dat')
        except FileNotFoundError:
            print('File "' + param['pkg_dir'] + 'forward_gas_mod/Data/wl_bins/' + param['wave_file'] + '.dat" not found. Using the native wavelength bins of opacities.')
            param['wl_native'] = True
            spectrum = np.loadtxt(param['pkg_dir'] + 'forward_gas_mod/Data/wl_bins/bins_02_20_R500.dat')

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


def cloud_pos(param):
    if param['fit_wtr_cld'] and param['fit_amm_cld']:
        pos_clda = int(find_nearest(param['P'], param['Pa_top']))

        if (param['clda_depth'] + param['P'][pos_clda]) > param['P'][-1]:
            param['clda_depth'] = param['P'][-1] - param['P'][pos_clda]

        pabot = int(find_nearest(param['P'], (param['clda_depth'] + param['P'][pos_clda])))

        depth_a = pabot - pos_clda
        if depth_a == 0:
            depth_a = 1

        pos_cldw = int(find_nearest(param['P'], (param['Pw_top'] + param['P'][pabot])))
        if (pos_cldw - pabot) == 0:
            pos_cldw += 1

        if pos_cldw >= len(param['P']):
            pos_cldw = len(param['P']) - 1

        if (param['cldw_depth'] + param['P'][pos_cldw]) > param['P'][-1]:
            param['cldw_depth'] = param['P'][-1] - param['P'][pos_cldw]

        pwbot = int(find_nearest(param['P'], (param['cldw_depth'] + param['P'][pos_cldw])))

        depth_w = pwbot - pos_cldw
        if depth_w == 0:
            depth_w = 1

    elif param['fit_wtr_cld'] and not param['fit_amm_cld']:
        pos_cldw = int(find_nearest(param['P'], param['Pw_top']))

        if (param['cldw_depth'] + param['P'][pos_cldw]) > param['P'][-1]:
            param['cldw_depth'] = param['P'][-1] - param['P'][pos_cldw]

        pwbot = int(find_nearest(param['P'], (param['cldw_depth'] + param['P'][pos_cldw])))

        depth_w = pwbot - pos_cldw
        if depth_w == 0:
            depth_w = 1

    elif param['fit_amm_cld'] and not param['fit_wtr_cld']:
        pos_clda = int(find_nearest(param['P'], param['Pa_top']))

        if (param['clda_depth'] + param['P'][pos_clda]) > param['P'][-1]:
            param['clda_depth'] = param['P'][-1] - param['P'][pos_clda]

        pabot = int(find_nearest(param['P'], (param['clda_depth'] + param['P'][pos_clda])))

        depth_a = pabot - pos_clda
        if depth_a == 0:
            depth_a = 1

    table = np.zeros((len(param['P']), 2))

    if param['fit_wtr_cld']:
        watermix = np.ones((len(param['P']))) * (param['CR_H2O'] * param['vmr_H2O'])
        dw = (np.log10(param['vmr_H2O']) - np.log10(param['CR_H2O'] * param['vmr_H2O'])) / depth_w
        for i in range(0, len(watermix)):
            if i <= pos_cldw:
                pass
            elif pos_cldw < i <= pos_cldw + depth_w:
                watermix[i] = 10. ** (np.log10(watermix[i - 1]) + dw)
            elif i > pos_cldw + depth_w:
                watermix[i] = watermix[i - 1]

        # watermix = np.loadtxt('/Users/mdamiano/packages/ExoReL/forward_mod/Result/Retrieval/watermix.dat')
        # watermix = np.loadtxt('/Users/mdamiano/Documents/Projects/Retrival/Code_gasplanet_ORIGINAL/Result/47Umab_real_meta1.5_opar3_Tint110_fhaze1.00e-36/watermix.dat')

        for i in range(0, len(watermix) - 1):
            if watermix[i] != watermix[i + 1]:
                table[i, 0] = float(1)
            else:
                pass
    else:
        watermix = np.ones((len(param['P']))) * param['vmr_H2O']

    if param['fit_amm_cld']:
        ammoniamix = np.ones((len(param['P']))) * (param['CR_NH3'] * param['vmr_NH3'])
        da = (np.log10(param['vmr_NH3']) - np.log10(param['CR_NH3'] * param['vmr_NH3'])) / depth_a
        for i in range(0, len(ammoniamix)):
            if i <= pos_clda:
                pass
            elif pos_clda < i <= pos_clda + depth_a:
                ammoniamix[i] = 10. ** (np.log10(ammoniamix[i - 1]) + da)
            elif i > pos_clda + depth_a:
                ammoniamix[i] = ammoniamix[i - 1]

        # ammoniamix = np.loadtxt('/Users/mdamiano/packages/ExoReL/forward_mod/Result/Retrieval/ammoniamix.dat')
        # ammoniamix = np.loadtxt('/Users/mdamiano/Documents/Projects/Retrival/Code_gasplanet_ORIGINAL/Result/47Umab_real_meta1.5_opar3_Tint110_fhaze1.00e-36/ammoniamix.dat')

        for i in range(0, len(ammoniamix) - 1):
            if ammoniamix[i] != ammoniamix[i + 1]:
                table[i, 1] = float(1)
            else:
                pass
    else:
        ammoniamix = np.ones((len(param['P']))) * param['vmr_NH3']

    param['watermix'] = watermix
    param['ammoniamix'] = ammoniamix
    param['cld_pos'] = table
    return param


def cloud_rocky_pos(param):
    if param['fit_wtr_cld']:
        # if param['Pw_top'] > param['P'][-1]:
        if param['Pw_top'] > param['P'][-1] or (param['Pw_top'] + param['cldw_depth']) > param['P'][-1]:
            no_cloud = True
        else:
            no_cloud = False

        if not no_cloud:
            pos_cldw = int(find_nearest(param['P_standard'], param['Pw_top']))

            if (param['cldw_depth'] + param['P_standard'][pos_cldw]) > param['P_standard'][-1]:
                param['cldw_depth'] = param['P_standard'][-1] - param['P_standard'][pos_cldw]

            pwbot = int(find_nearest(param['P_standard'], (param['cldw_depth'] + param['P_standard'][pos_cldw])))

            depth_w = pwbot - pos_cldw
            if depth_w == 0:
                return np.ones((len(param['P']))) * param['vmr_H2O']
            else:
                pass

            watermix = np.ones((len(param['P_standard']))) * (param['CR_H2O'] * param['vmr_H2O'])
            dw = (np.log10(param['vmr_H2O']) - np.log10(param['CR_H2O'] * param['vmr_H2O'])) / depth_w
            for i in range(0, len(watermix)):
                if i <= pos_cldw:
                    pass
                elif pos_cldw < i <= pos_cldw + depth_w:
                    watermix[i] = 10. ** (np.log10(watermix[i - 1]) + dw)
                elif i > pos_cldw + depth_w:
                    watermix[i] = watermix[i - 1]
            watermix = watermix[:len(param['P'])]
        else:
            watermix = np.ones((len(param['P']))) * param['vmr_H2O']
    else:
        watermix = np.ones((len(param['P']))) * param['vmr_H2O']

    return watermix
    # return gaussian_filter1d(watermix, 1, mode='nearest')


def adjust_VMR(param, all_gases=True):
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
            param['vmr_' + mol] = np.zeros(len(param['vmr_H2O']))

        for i in range(0, len(param['vmr_H2O'])):
            res[-1] = 1.0 - param['vmr_H2O'][i]
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

        v_m_r = np.zeros(len(param['vmr_H2O']))
        for mol in param['fit_molecules']:
            if mol == 'H2O' or mol == considered_fill:
                pass
            else:
                param['vmr_' + mol] = np.ones(len(param['vmr_H2O'])) * param['vmr_' + mol]
                v_m_r += param['vmr_' + mol]

        param['vmr_' + considered_fill] = np.ones(len(param['vmr_H2O'])) - v_m_r - param['vmr_H2O']

    return param


def ozone_earth_mask(param):
    otop, obot = (10. ** 1.5), (10. ** 4.0)
    idxs_top = np.where(otop > param['P'])[0]
    idxs_bot = np.where(param['P'] > obot)[0]
    param['vmr_O3'][idxs_top] = 10. ** (-12.0)
    param['vmr_O3'][idxs_bot] = 10. ** (-12.0)

    return param['vmr_O3']


def ranges(param):
    if not param['rocky']:
        param['ch2o_range'] = [-12.0, 0.0]              # concentration of H2O
        param['cnh3_range'] = [-12.0, 0.0]              # concentration of NH3
        param['cch4_range'] = [-12.0, 0.0]              # concentration of CH4
        if param['fit_amm_cld']:
            param['ptopa_range'] = [0.0, 8.0]           # Top pressure NH3
            param['dclda_range'] = [0.0, 8.5]           # Depth NH3 cloud
            param['crnh3_range'] = [-12.0, 0.0]         # Condensation Ratio NH3
    else:
        if param['fit_p0'] and param['gas_par_space'] != 'partial_pressure':
            param['p0_range'] = [4.5, 7.0]             # Surface pressure

        for mol in param['fit_molecules']:
            if (param['gas_par_space'] == 'centered_log_ratio' or param['gas_par_space'] == 'clr') and not param['mod_prior']:
                param['clr' + mol + '_range'] = [-25.0, 25.0]  # centered-log-ratio ranges
            elif param['gas_par_space'] == 'volume_mixing_ratio' or param['gas_par_space'] == 'vmr':
                param['vmr' + mol + '_range'] = [-10.0, 0.0]  # volume mixing ratio ranges
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
                param['ag_x2_range'] = [0.01, 0.5]

        if param['fit_T']:
            param['tp_range'] = [0.0, 700.0]            # Atmospheric equilibrium temperature

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

        if param['fit_p_size'] and param['p_size_type'] == 'constant':
            param['p_size_range'] = [-1.0, 2.0]
        elif param['fit_p_size'] and param['p_size_type'] == 'factor':
            param['p_size_range'] = [-1.0, 1.0]
        else:
            pass

        if param['fit_cld_frac']:
            param['cld_frac_range'] = [-3.0, 0.0]

    if param['fit_wtr_cld']:
        param['ptopw_range'] = [2.0, 7.0]               # Top pressure H2O
        param['dcldw_range'] = [2.0, 7.0]               # Depth H2O cloud
        param['crh2o_range'] = [-7.0, 0.0]             # Condensation Ratio H2O
    if param['fit_g']:
        param['gp_range'] = [1.0, 6.0]                  # Gravity
    if param['fit_phi']:
        param['phi_range'] = [0.0, 180.0]               # Phase Angle

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
    except KeyError:
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
        new_sp.append(float(trapz(np.array([sp[i], sp[i + 1]]), x=np.array([wl[i], wl[i + 1]]))))

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
        # Solar Spectrum
        data = np.loadtxt(param['pkg_dir'] + 'forward_gas_mod/PlanetModel/RayleightOpa/solar.txt')
        tck = interp1d(data[:, 0], data[:, 1])
        param['solar'] = tck(param['wavelength_planet'])

        # Methane Opacity
        data = np.loadtxt(param['pkg_dir'] + 'forward_gas_mod/PlanetModel/MethaneOpa/CH4cross.txt')
        tck = interp1d(data[:, 0], data[:, 1])
        param['crossCH4'] = tck(param['wavelength_planet'])

        # Ammonia Opacity
        data = np.loadtxt(param['pkg_dir'] + 'forward_gas_mod/PlanetModel/MethaneOpa/NH3cross.txt')
        tck = interp1d(data[:, 0], data[:, 1])
        param['crossNH3'] = tck(param['wavelength_planet'])

        # Water Opacity
        data = np.loadtxt(param['pkg_dir'] + 'forward_gas_mod/PlanetModel/MethaneOpa/H2Ocross.txt')
        tck = interp1d(data[:, 0], data[:, 1])
        param['crossH2O'] = tck(param['wavelength_planet'])

        # E2
        param['E2'] = np.loadtxt(param['pkg_dir'] + 'forward_gas_mod/PlanetModel/E2.dat')

        #    cloud output
        data = np.loadtxt(param['pkg_dir'] + 'forward_gas_mod/PlanetModel/CrossP/cross_H2OLiquid_M.dat')
        param['H2OL_r'] = data[:, 0]  # zero-order radius, in micron
        param['H2OL_c'] = data[:, 1:]  # cross-section per droplet, in cm2
        data = np.loadtxt(param['pkg_dir'] + 'forward_gas_mod/PlanetModel/CrossP/albedo_H2OLiquid_M.dat')
        param['H2OL_a'] = data[:, 1:]
        data = np.loadtxt(param['pkg_dir'] + 'forward_gas_mod/PlanetModel/CrossP/geo_H2OLiquid_M.dat')
        param['H2OL_g'] = data[:, 1:]

        data = np.loadtxt(param['pkg_dir'] + 'forward_gas_mod/PlanetModel/CrossP/cross_H2OIce_M.dat')
        param['H2OI_r'] = data[:, 0]
        param['H2OI_c'] = data[:, 1:]
        data = np.loadtxt(param['pkg_dir'] + 'forward_gas_mod/PlanetModel/CrossP/albedo_H2OIce_M.dat')
        param['H2OI_a'] = data[:, 1:]
        data = np.loadtxt(param['pkg_dir'] + 'forward_gas_mod/PlanetModel/CrossP/geo_H2OIce_M.dat')
        param['H2OI_g'] = data[:, 1:]

        data = np.loadtxt(param['pkg_dir'] + 'forward_gas_mod/PlanetModel/CrossP/cross_NH3Ice_M.dat')
        param['NH3I_r'] = data[:, 0]
        param['NH3I_c'] = data[:, 1:]
        data = np.loadtxt(param['pkg_dir'] + 'forward_gas_mod/PlanetModel/CrossP/albedo_NH3Ice_M.dat')
        param['NH3I_a'] = data[:, 1:]
        data = np.loadtxt(param['pkg_dir'] + 'forward_gas_mod/PlanetModel/CrossP/geo_NH3Ice_M.dat')
        param['NH3I_g'] = data[:, 1:]

        # MH0-RM
        data = np.genfromtxt(param['pkg_dir'] + 'forward_gas_mod/PlanetModel/MeanOpacity/MH0-RM.txt', delimiter='\t', dtype=float)
        param['t_g'] = data[0, 1:]
        param['p_g'] = data[1:, 0]
        param['r_s'] = data[1:, 1:]

    elif param['rocky']:
        # Mass-Radius diagram
        if param['fit_Mp'] and param['fit_Rp']:
            if param['Rp_prior_type'] == 'R_M_prior':
                M_R_Fe = np.loadtxt(param['pkg_dir'] + 'forward_rocky_mod/Data/Fe_mass_radius_jup.dat')
                M_R_H2O = np.loadtxt(param['pkg_dir'] + 'forward_rocky_mod/Data/H2O_mass_radius_jup.dat')
                param['R-M_Fe'] = interp1d(M_R_Fe[:, 1], M_R_Fe[:, 0])
                param['R-M_H2O'] = interp1d(M_R_H2O[:, 1], M_R_H2O[:, 0])
            elif param['Mp_prior_type'] == 'M_R_prior':
                M_R_Fe = np.loadtxt(param['pkg_dir'] + 'forward_rocky_mod/Data/Fe_mass_radius_jup.dat')
                M_R_H2O = np.loadtxt(param['pkg_dir'] + 'forward_rocky_mod/Data/H2O_mass_radius_jup.dat')
                param['M-R_Fe'] = interp1d(M_R_Fe[:, 0], M_R_Fe[:, 1])
                param['M-R_H2O'] = interp1d(M_R_H2O[:, 0], M_R_H2O[:, 1])
            else:
                pass
        #    Load Mie Calculation Results
        data = np.loadtxt(param['pkg_dir'] + 'forward_rocky_mod/PlanetModel/CrossP/Cross_water_wavelength.dat')
        param['H2OL_r'] = data[:, 0]  # zero-order radius, in micron
        param['H2OL_c'] = data[:, 1:]  # cross section per droplet, in cm2
        data = np.loadtxt(param['pkg_dir'] + 'forward_rocky_mod/PlanetModel/CrossP/Albedo_water_wavelength.dat')
        param['H2OL_a'] = data[:, 1:]
        data = np.loadtxt(param['pkg_dir'] + 'forward_rocky_mod/PlanetModel/CrossP/Geo_water_wavelength.dat')
        param['H2OL_g'] = data[:, 1:]

    return param


def retrieval_par_and_npar(param):
    if not param['rocky']:
        parameters = ["Log(H$_2$O)", "Log(NH$_3$)", "Log(CH$_4$)"]
        if param['fit_wtr_cld']:
            parameters.append("Log(P$_{top, H_2O}$)")
            parameters.append("Log(D$_{H_2O}$)")
            parameters.append("Log(CR$_{H_2O}$)")
        if param['fit_amm_cld']:
            parameters.append("Log(P$_{top, NH_3}$)")
            parameters.append("Log(D$_{NH_3}$)")
            parameters.append("Log(CR$_{NH_3}$)")
        if param['fit_g']:
            parameters.append("Log(g)")
        if param['obs_numb'] is None:
            if param['fit_phi']:
                parameters.append("$\phi$")
        else:
            for obs in range(0, param['obs_numb']):
                if param['fit_phi']:
                    parameters.append("$\phi_" + str(obs) + "$")

    else:
        parameters = []
        if param['fit_p0']:
            parameters.append("P$_0$")
        if param['fit_wtr_cld']:
            parameters.append("Log(P$_{top, H_2O}$)")
            parameters.append("Log(D$_{H_2O}$)")
            parameters.append("Log(CR$_{H_2O}$)")
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
            parameters.append("T$_p$")
        if param['fit_cld_frac']:
            parameters.append("cld frac")
        if param['fit_g']:
            parameters.append("Log(g)")
        if param['fit_Mp']:
            parameters.append("M$_p$")
        if param['fit_Rp']:
            parameters.append("R$_p$")
        if param['fit_p_size']:
            parameters.append("P$_{size}$")
        if param['fit_phi']:
            if param['obs_numb'] is None:
                parameters.append("$\phi$")
            else:
                for obs in range(0, param['obs_numb']):
                    parameters.append("$\phi_" + str(obs) + "$")

    return parameters, len(parameters)


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
            param['beta'] += 0.0
        except KeyError:
            param['beta'] = param['snr'] / SNR[i0]
        SNR *= param['beta']

        # convert SNR to error
        err = contrast / SNR

        try:
            SNR = np.array([param['spectrum']['wl'], SNR]).T
            if param['spectrum']['bins']:
                SNR = np.concatenate((np.array([param['spectrum']['wl_high']]).T, SNR), axis=1)
                SNR = np.concatenate((np.array([param['spectrum']['wl_low']]).T, SNR), axis=1)
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
    file_list = glob.glob(directory + 'forward_rocky_mod/core_*.c')
    if len(file_list) > 0:
        for i in file_list:
            os.system('rm ' + i)

    file_list = glob.glob(directory + 'forward_rocky_mod/par_*.h')
    if len(file_list) > 0:
        for i in file_list:
            os.system('rm ' + i)

    file_list = glob.glob(directory + 'forward_rocky_mod/None*')
    if len(file_list) > 0:
        for i in file_list:
            os.system('rm ' + i)

    for j in range(0, 10):
        file_list = glob.glob(directory + 'forward_rocky_mod/' + str(j) + '*')
        if len(file_list) > 0:
            for i in file_list:
                os.system('rm ' + i)

    file_list = glob.glob(directory + 'forward_rocky_mod/Result/Retrieval_*')
    if len(file_list) > 0:
        for i in file_list:
            os.system('rm -rf ' + i)
