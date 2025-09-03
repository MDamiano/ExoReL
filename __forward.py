from .__basics import *
from .__utils import *


class FORWARD_MODEL:
    def __init__(self, param, retrieval=True, canc_metadata=False):
        self.param = copy.deepcopy(param)
        self.process = str(self.param['core_number']) + str(random.randint(0, 100000)) + alphabet() + alphabet() + alphabet() + str(random.randint(0, 100000))
        self.package_dir = param['pkg_dir']
        self.retrieval = retrieval
        self.canc_metadata = canc_metadata
        self.hazes_calc = param['hazes']
        self.c_code_directory = self.package_dir + 'forward_mod/'
        self.matlab_code_directory = self.c_code_directory + 'PlanetModel/'
        try:
            self.working_dir = param['wkg_dir']
        except KeyError:
            self.working_dir = os.getcwd()

    def __surface_structure(self):
        if self.param['fit_ag']:
            self.surf_alb = np.ones(len(self.param['wl_C_grid']))
            if self.param['surface_albedo_parameters'] == int(1):
                self.surf_alb *= self.param['Ag']
            elif self.param['surface_albedo_parameters'] == int(3):
                x1_indx = np.where(self.param['wl_C_grid'] < self.param['Ag_x1'])[0]
                self.surf_alb[x1_indx] *= self.param['Ag1']
                self.surf_alb[x1_indx[-1] + 1:] *= self.param['Ag2']
            elif self.param['surface_albedo_parameters'] == int(5):
                x1_indx = np.where(self.param['wl_C_grid'] < self.param['Ag_x1'])[0]
                x2_indx = np.where((self.param['wl_C_grid'] > self.param['Ag_x1']) & (self.param['wl_C_grid'] < self.param['Ag_x2']))[0]
                self.surf_alb[x1_indx] *= self.param['Ag1']
                self.surf_alb[x2_indx] *= self.param['Ag2']
                self.surf_alb[x2_indx[-1] + 1:] *= self.param['Ag3']
        else:
            if self.param['Ag'] is not None:
                self.surf_alb = self.param['Ag'] * np.ones(len(self.param['wl_C_grid']))
            else:
                self.surf_alb = np.zeros(len(self.param['wl_C_grid']))

        with open(self.outdir + 'surface_albedo.dat', 'w') as file:
            for i in range(0, len(self.surf_alb)):
                file.write("{:.6e}".format(self.surf_alb[i]))
                file.write('\n')

    def __atmospheric_structure(self):
        try:
            os.mkdir(self.c_code_directory + 'Result/Retrieval_' + str(self.process) + '/')
        except OSError:
            self.process = alphabet() + str(random.randint(0, 100000))
            os.mkdir(self.c_code_directory + 'Result/Retrieval_' + str(self.process) + '/')

        self.outdir = self.c_code_directory + 'Result/Retrieval_' + str(self.process) + '/'

        deltaP = 0.001 * self.__waterpressure(220)  # assume super saturation to be 0.1% at 220 K

        g = self.param['gp'] + 0.0

        # Set up pressure grid
        P = self.param['P'] + 0.0  # in Pascal

        # Temperature profile (isothermal)
        T = self.param['Tp'] * np.ones(len(P))

        # Cloud density calculation
        cloudden = 1.0e-36 * np.ones(len(P))
        for i in range(len(P) - 2, -1, -1):
            cloudden[i] = max(abs(self.param['vmr_H2O'][i] - self.param['vmr_H2O'][i + 1]) * 0.018 * P[i] / const.R.value / T[i], 1e-25)  # kg/m^3, g/L

        # Particle size calculation
        particlesize = 1.0e-36 * np.ones(len(P))
        if self.param['fit_p_size'] and self.param['p_size_type'] == 'constant':
            particlesize = self.param['p_size'] * np.ones(len(P))
        else:
            for i in range(len(P) - 2, -1, -1):
                r0, r1, r2, VP = particlesizef(g, T[i], P[i], self.param['mean_mol_weight'][i], self.param['mm']['H2O'], self.param['KE'], deltaP)
                if self.param['fit_p_size'] and self.param['p_size_type'] == 'factor':
                    particlesize[i] = r2 * self.param['p_size']
                else:
                    particlesize[i] = r2 + 0.0

        # Calculate the height
        P = P[::-1]
        T = T[::-1]
        cloudden = cloudden[::-1]
        particlesize = particlesize[::-1]
        MMM = self.param['mean_mol_weight'][::-1]

        # Atmospheric Composition
        f = {}
        for mol in self.param['fit_molecules']:
            f[mol] = self.param['vmr_' + mol][::-1]
        if self.param['gas_fill'] is not None:
            f[self.param['gas_fill']] = self.param['vmr_' + self.param['gas_fill']][::-1]

        Z = np.zeros(len(P))
        for j in range(0, len(P) - 1):
            H = const.k_B.value * (T[j] + T[j + 1]) / 2. / g / MMM[j] / const.u.value / 1000.  # km
            Z[j + 1] = Z[j] + H * np.log(P[j] / P[j + 1])

        # Adaptive grid
        if self.param['use_adaptive_grid']:
            idx_cloud_layers = np.where(np.diff(f['H2O']) != 0.0)[0] + 1
            if len(idx_cloud_layers) > 0:
                n_cloud_layers = int(round((self.param['n_layer'] + 1) / 3, 0))
                n_above_layers = int(round((self.param['n_layer'] + 1 - n_cloud_layers) / 2, 0))
                n_below_layers = (self.param['n_layer'] + 1) - n_cloud_layers - n_above_layers

                Z_below = np.linspace(Z[0], Z[min(idx_cloud_layers) - 1], num=n_below_layers, endpoint=False)
                Z_within = np.linspace(Z[min(idx_cloud_layers) - 1], Z[max(idx_cloud_layers)], num=n_cloud_layers, endpoint=False)
                Z_above = np.linspace(Z[max(idx_cloud_layers)], Z[-1], num=n_above_layers, endpoint=True)
                zz = np.concatenate((np.concatenate((Z_below, Z_within)), Z_above))
            else:
                zz = np.linspace(Z[0], Z[-1], num=int(self.param['n_layer'] + 1), endpoint=True)
        else:
            zz = np.linspace(Z[0], Z[-1], num=int(self.param['n_layer'] + 1), endpoint=True)

        if not self.retrieval:
            np.savetxt(self.outdir + 'watermix.dat', f['H2O'])

            np.savetxt(self.outdir + 'particlesize.dat', particlesize)

            np.savetxt(self.outdir + 'cloudden.dat', cloudden)

            np.savetxt(self.outdir + 'P.dat', P)
            np.savetxt(self.outdir + 'T.dat', T)

        z0 = zz[:-1]
        z1 = zz[1:]
        zl = np.mean([z0, z1], axis=0)
        tck = interp1d(Z, T)
        tl = tck(zl)
        tck = interp1d(Z, np.log(P))
        pl = np.exp(tck(zl))

        nden = pl / const.k_B.value / tl * 1.0E-6  # molecule cm-3
        n = {}
        for mol in self.param['fit_molecules']:
            tck = interp1d(Z, np.log(f[mol]))
            n[mol] = np.exp(tck(zl)) * nden
        if self.param['gas_fill'] is not None:
            tck = interp1d(Z, np.log(f[self.param['gas_fill']]))
            n[self.param['gas_fill']] = np.exp(tck(zl)) * nden

        tck = interp1d(Z, np.log(cloudden))
        cloudden = np.exp(tck(zl))

        tck = interp1d(Z, np.log(particlesize))
        particlesize = np.exp(tck(zl))

        #    Generate ConcentrationSTD.dat file
        NSP = 111
        with open(self.outdir + 'ConcentrationSTD.dat', 'w') as file:
            file.write('z\t\tz0\t\tz1\t\tT\t\tP')
            for i in range(1, NSP + 1):
                file.write('\t\t' + str(i))
            file.write('\n')
            file.write('km\t\tkm\t\tkm\t\tK\t\tPa\n')
            for j in range(0, len(zl)):
                file.write("{:.6f}".format(zl[j]) + '\t\t' + "{:.6f}".format(z0[j]) + '\t\t' + "{:.6f}".format(z1[j]) + '\t\t' + "{:.6f}".format(tl[j]) + '\t\t' + "{:.6e}".format(pl[j]))
                for i in range(1, NSP + 1):
                    # H2O
                    if i == 7 and 'H2O' in self.param['fit_molecules']:
                        if self.param['contribution'] and self.param['mol_contr'] == 'H2O':
                            file.write('\t\t' + "{:.6e}".format(n['H2O'][j]))
                        elif self.param['contribution'] and self.param['mol_contr'] != 'H2O':
                            file.write('\t\t' + "{:.6e}".format(0.0))
                        else:
                            file.write('\t\t' + "{:.6e}".format(n['H2O'][j]))

                    # NH3
                    elif i == 9 and 'NH3' in self.param['fit_molecules']:
                        if self.param['contribution'] and self.param['mol_contr'] == 'NH3':
                            file.write('\t\t' + "{:.6e}".format(n['NH3'][j]))
                        elif self.param['contribution'] and self.param['mol_contr'] != 'NH3':
                            file.write('\t\t' + "{:.6e}".format(0.0))
                        else:
                            file.write('\t\t' + "{:.6e}".format(n['NH3'][j]))

                    # CH4
                    elif i == 21 and 'CH4' in self.param['fit_molecules']:
                        if self.param['contribution'] and self.param['mol_contr'] == 'CH4':
                            file.write('\t\t' + "{:.6e}".format(n['CH4'][j]))
                        elif self.param['contribution'] and self.param['mol_contr'] != 'CH4':
                            file.write('\t\t' + "{:.6e}".format(0.0))
                        else:
                            file.write('\t\t' + "{:.6e}".format(n['CH4'][j]))

                    # SO2
                    elif i == 43 and 'SO2' in self.param['fit_molecules']:
                        if self.param['contribution'] and self.param['mol_contr'] == 'SO2':
                            file.write('\t\t' + "{:.6e}".format(n['SO2'][j]))
                        elif self.param['contribution'] and self.param['mol_contr'] != 'SO2':
                            file.write('\t\t' + "{:.6e}".format(0.0))
                        else:
                            file.write('\t\t' + "{:.6e}".format(n['SO2'][j]))

                    # H2S
                    elif i == 45 and 'H2S' in self.param['fit_molecules']:
                        if self.param['contribution'] and self.param['mol_contr'] == 'H2S':
                            file.write('\t\t' + "{:.6e}".format(n['H2S'][j]))
                        elif self.param['contribution'] and self.param['mol_contr'] != 'H2S':
                            file.write('\t\t' + "{:.6e}".format(0.0))
                        else:
                            file.write('\t\t' + "{:.6e}".format(n['H2S'][j]))

                    # CO2
                    elif i == 52 and 'CO2' in self.param['fit_molecules']:
                        if self.param['contribution'] and self.param['mol_contr'] == 'CO2':
                            file.write('\t\t' + "{:.6e}".format(n['CO2'][j]))
                        elif self.param['contribution'] and self.param['mol_contr'] != 'CO2':
                            file.write('\t\t' + "{:.6e}".format(0.0))
                        else:
                            file.write('\t\t' + "{:.6e}".format(n['CO2'][j]))

                    # CO
                    elif i == 20 and 'CO' in self.param['fit_molecules']:
                        if self.param['contribution'] and self.param['mol_contr'] == 'CO':
                            file.write('\t\t' + "{:.6e}".format(n['CO'][j]))
                        elif self.param['contribution'] and self.param['mol_contr'] != 'CO':
                            file.write('\t\t' + "{:.6e}".format(0.0))
                        else:
                            file.write('\t\t' + "{:.6e}".format(n['CO'][j]))

                    # O2
                    elif i == 54 and 'O2' in self.param['fit_molecules']:
                        if self.param['contribution'] and self.param['mol_contr'] == 'O2':
                            file.write('\t\t' + "{:.6e}".format(n['O2'][j]))
                        elif self.param['contribution'] and self.param['mol_contr'] != 'O2':
                            file.write('\t\t' + "{:.6e}".format(0.0))
                        else:
                            file.write('\t\t' + "{:.6e}".format(n['O2'][j]))

                    # O3
                    elif i == 2 and 'O3' in self.param['fit_molecules']:
                        if self.param['contribution'] and self.param['mol_contr'] == 'O3':
                            file.write('\t\t' + "{:.6e}".format(n['O3'][j]))
                        elif self.param['contribution'] and self.param['mol_contr'] != 'O3':
                            file.write('\t\t' + "{:.6e}".format(0.0))
                        else:
                            file.write('\t\t' + "{:.6e}".format(n['O3'][j]))

                    # N2O
                    elif i == 11 and 'N2O' in self.param['fit_molecules']:
                        if self.param['contribution'] and self.param['mol_contr'] == 'N2O':
                            file.write('\t\t' + "{:.6e}".format(n['N2O'][j]))
                        elif self.param['contribution'] and self.param['mol_contr'] != 'N2O':
                            file.write('\t\t' + "{:.6e}".format(0.0))
                        else:
                            file.write('\t\t' + "{:.6e}".format(n['N2O'][j]))

                    # N2
                    elif i == 55 and 'N2' in self.param['fit_molecules']:
                        if self.param['contribution'] and self.param['mol_contr'] == 'N2':
                            file.write('\t\t' + "{:.6e}".format(n['N2'][j]))
                        elif self.param['contribution'] and self.param['mol_contr'] != 'N2':
                            file.write('\t\t' + "{:.6e}".format(0.0))
                        else:
                            file.write('\t\t' + "{:.6e}".format(n['N2'][j]))
                    elif i == 55 and self.param['gas_fill'] == 'N2':
                        if self.param['contribution'] and self.param['mol_contr'] == 'N2':
                            file.write('\t\t' + "{:.6e}".format(n['N2'][j]))
                        elif self.param['contribution'] and self.param['mol_contr'] != 'N2':
                            file.write('\t\t' + "{:.6e}".format(0.0))
                        else:
                            file.write('\t\t' + "{:.6e}".format(n['N2'][j]))

                    # H2
                    elif i == 53 and self.param['gas_fill'] == 'H2':
                        if self.param['contribution'] and self.param['mol_contr'] == 'H2':
                            file.write('\t\t' + "{:.6e}".format(n['H2'][j]))
                        elif self.param['contribution'] and self.param['mol_contr'] != 'H2':
                            file.write('\t\t' + "{:.6e}".format(0.0))
                        else:
                            file.write('\t\t' + "{:.6e}".format(n['H2'][j]))
                    else:
                        file.write('\t\t' + "{:.6e}".format(0.0))
                file.write('\n')

        #    cloud output
        crow = np.zeros((len(zl), 324))
        albw = np.ones((len(zl), 324))
        geow = np.zeros((len(zl), 324))

        #    opacity
        sig = 2
        for j in range(0, len(zl)):
            r2 = particlesize[j]
            if cloudden[j] < 1e-16:
                pass
            else:
                r0 = r2 * np.exp(-np.log(sig) ** 2.)
                VP = 4. * math.pi / 3. * ((r2 * 1.0e-6 * np.exp(0.5 * np.log(sig) ** 2.)) ** 3.) * 1.0e+6 * 1.0  # g
                for indi in range(0, 324):
                    tck = interp1d(np.log10(self.param['H2OL_r']), np.log10(self.param['H2OL_c'][:, indi]))
                    temporaneo = tck(np.log10(max(0.01, min(r0, 100))))
                    crow[j, indi] = cloudden[j] / VP * 1.0e-3 * (10. ** temporaneo)  # cm-1
                    tck = interp1d(np.log10(self.param['H2OL_r']), self.param['H2OL_a'][:, indi])
                    albw[j, indi] = tck(np.log10(max(0.01, min(r0, 100))))
                    tck = interp1d(np.log10(self.param['H2OL_r']), self.param['H2OL_g'][:, indi])
                    geow[j, indi] = tck(np.log10(max(0.01, min(r0, 100))))

        with open(self.outdir + 'cross_H2O.dat', 'w') as file:
            for j in range(0, len(zl)):
                for indi in range(0, 324):
                    file.write("{:.6e}".format(crow[j, indi]) + '\t')
                file.write('\n')

        with open(self.outdir + 'albedo_H2O.dat', 'w') as file:
            for j in range(0, len(zl)):
                for indi in range(0, 324):
                    file.write("{:.6e}".format(albw[j, indi]) + '\t')
                file.write('\n')

        with open(self.outdir + 'geo_H2O.dat', 'w') as file:
            for j in range(0, len(zl)):
                for indi in range(0, 324):
                    file.write("{:.6e}".format(geow[j, indi]) + '\t')
                file.write('\n')

    def __waterpressure(self, t):
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
                # Formulation] from Seinfeld & Pandis(2006)
                a = 1 - (373.15 / t[i])
                p[i] = 101325. * np.exp((13.3185 * a) - (1.97 * (a ** 2.)) - (0.6445 * (a ** 3.)) - (0.1229 * (a ** 4.)))
            elif t[i] < 647.09:
                p[i] = (10. ** (8.14019 - (1810.94 / (244.485 + t[i] - 273.15)))) * 133.322387415
            else:
                p[i] = np.nan
        return p

    def __run_structure(self):
        os.chdir(self.matlab_code_directory)
        self.__atmospheric_structure()
        self.__surface_structure()
        os.chdir(self.working_dir)

    def __par_c_file(self):
        c_par_file = ['#ifndef _PLANET_H_\n',
                      '#define _PLANET_H_\n',
                      # Planet Physical Properties
                      # '#define MASS_PLANET          ' + str(self.param['Mp'] * const.M_jup.value) + '\n',  # kg
                      # '#define RADIUS_PLANET        ' + str(self.param['Rp'] * const.R_jup.value) + '\n',  # m

                      # Planet Orbital Properties
                      '#define ORBIT                ' + str(self.param['equivalent_a']) + '\n',  # AU
                      '#define STAR_SPEC            "Data/solar0.txt"\n',
                      '#define SURF_SPEC            "Result/Retrieval_' + str(self.process) + '/surface_albedo.dat"\n',
                      '#define TIDELOCK             0\n',  # If the planet is tidally locked
                      '#define FaintSun             1.0\n',  # Faint early Sun factor
                      '#define STAR_TEMP            ' + str(self.param['Tirr']) + '\n',  # 394.109 irradiation Temperature at 1 AU
                      '#define THETAREF             1.0471\n',  # Slant Path Angle in radian
                      '#define PAB                  0.343\n',  # Planet Bond Albedo
                      '#define FADV                 0.25\n',  # Advection factor: 0.25=uniformly distributed, 0.6667=no Advection
                      # '#define PSURFAB              ' + str(float(self.param['Ag'])) + '\n',  # Planet Surface Albedo
                      '#define PSURFEM              0.0\n',  # Planet Surface Emissivity
                      '#define DELADJUST            1\n',  # Whether use the delta adjustment in the 2-stream diffuse radiation
                      '#define TAUTHRESHOLD         0.1\n',  # Optical Depth Threshold for multi-layer diffuse radiation
                      '#define TAUMAX               1000.0\n',  # Maximum optical Depth in the diffuse radiation
                      '#define TAUMAX1              1000.0\n',  # Maximum optical Depth in the diffuse radiation
                      '#define TAUMAX2              1000.0\n',
                      '#define IFDIFFUSE            1\n',  # Set to 1 if want to include diffuse solar radiation into the photolysis rate
                      #
                      '#define IFUVMULT             0\n',  # Whether do the UV Multiplying
                      '#define FUVMULT              1.0E+3\n',  # Multiplying factor for FUV radiation <200 nm
                      '#define MUVMULT              1.0E+2\n',  # Multiplying factor for MUV radiation 200 - 300 nm
                      '#define NUVMULT              1.0E+1\n',  # Multiplying factor for NUV radiation 300 - 400 nm
                      #
                      # Planet Temperature-Pressure Preofile
                      '#define TPMODE               1\n',  # 1: import data from a ZTP list
                      #                                      0: calculate TP profile from the parametized formula*/
                      '#define TPLIST               "Data/TP1986.dat"\n',
                      '#define PTOP                 1.0E-5\n',  # Pressure at the top of atmosphere in bar
                      '#define TTOP				    ' + str(self.param['Tp']) + '\n',  # Temperature at the top of atmosphere
                      '#define TSTR                 ' + str(self.param['Tp']) + '\n',  # Temperature at the top of stratosphere
                      '#define TINV                 0\n',  # set to 1 if there is a temperature inversion
                      '#define PSTR                 1.0E-1\n',  # Pressure at the top of stratosphere
                      '#define PMIDDLE				0\n',  # Pressure at the bottom of stratosphere
                      '#define TMIDDLE				' + str(self.param['Tp']) + '\n',  # Temperature at the bottom of stratosphere
                      '#define PBOTTOM				1.0E+0\n',  # Pressure at the bottom of stratosphere
                      '#define TBOTTOM				' + str(self.param['Tp']) + '\n',  # Temperature at the bottom of stratosphere
                      '#define PPOFFSET			    0.0\n',  # Pressure offset in log [Pa]
                      #
                      # Calculation Grids
                      # '#define zbin                 180\n',  # How many altitude bin?
                      '#define zbin                 ' + str(int(self.param['n_layer'])) + '\n',  # How many altitude bin?
                      # '#define zmax                 1631.0\n',  # Maximum altitude in km
                      # '#define zmin                 0.0\n',  # Maximum altitude in km
                      '#define WaveBin              9999\n',  # How many wavelength bin?
                      '#define WaveMin              1.0\n',  # Minimum Wavelength in nm
                      '#define WaveMax              10000.0\n',  # Maximum Wavelength in nm
                      '#define WaveMax1             1000.0\n',  # Maximum Wavelength in nm for the Calculation of UV-visible radiation and photolysis rates
                      '#define TDEPMAX	            300.0\n',  # Maximum Temperature-dependence Validity for UV Cross sections
                      '#define TDEPMIN              200.0\n',  # Minimum Temperature-dependence Validity for UV Cross sections

                      # The criteria of convergence
                      '#define Tol1                 1.0E+10\n',
                      '#define Tol2                 1.0E-16\n',
                      #
                      # Mode of iteration
                      '#define TSINI                1.0E-18\n',  # Initial Trial Timestep, generally 1.0E-8
                      '#define FINE1                1\n',  # Set to one for fine iteration: Set to 2 to disregard the bottom boundary layers
                      '#define FINE2                1\n',  # Set to one for fine iteration: Set to 2 to disregard the fastest varying point
                      '#define TMAX                 1.0E+12\n',  # Maximum of time step
                      '#define TMIN                 1.0E-25\n',  # Minimum of time step
                      '#define TSPEED               1.0E+12\n',  # Speed up factor
                      '#define NMAX                 1E+4\n',  # Maximum iteration cycles
                      '#define NMAXT                1.0E+13\n',  # Maximum iteration cumulative time in seconds
                      '#define MINNUM               1.0E-0\n',  # Minimum number density in denominator
                      #
                      # Molecular Species
                      '#define NSP                  111\n',  # Number of species in the standard list
                      '#define SPECIES_LIST         "Data/species_Earth_Full.dat"\n',
                      # '#define AIRM_FILE            "Result/Retrieval_' + str(self.process) + '/mean_mol_mass.dat"\n',
                      '#define AIRM                 ' + str(self.param['mean_mol_weight'][-1]) + '\n',  # Initial mean molecular mass of atmosphere, in atomic mass unit
                      '#define AIRVIS               1.0E-5\n',  # Dynamic viscosity in SI
                      # '#define RefIdxType           0\n',  # Type of Refractive Index: 0=Air, 1=CO2, 2=He, 3=N2, 4=NH3, 5=CH4, 6=H2, 7=O2, 8=composition
                      #
                      # Aerosol Species
                      '#define AERSIZE              1.0E-7\n',  # diameter in m
                      '#define AERDEN               1.84E+3\n',  # density in SI
                      '#define NCONDEN              1\n',  # Calculate the condensation every NCONDEN iterations
                      '#define IFGREYAER            0\n',  # Contribute to the grey atmosphere Temperature? 0=no, 1=yes
                      '#define SATURATIONREDUCTION  1.0\n',  # Ad hoc reduction factor for saturation pressure of water
                      '#define AERRADFILE1          "Data/H2SO4AER_CrossM_01.dat"\n',  # radiative properties of H2SO4
                      '#define AERRADFILE2          "Data/S8AER_CrossM_01.dat"\n',  # radiative properties of S8
                      #
                      # Initial Concentration Setting
                      '#define IMODE                4\n',  # 1: Import from SPECIES_LIST
                      #                                    # 0: Calculate initial concentrations from chemical equilibrium sub-routines (not rad)
                      #                                    # 3: Calculate initial concentrations from simplied chemical equilibrium formula (not rad)
                      #                                    # 2: Import from results of previous calculations
                      #                                    # 4: Import from results of previous calculations in the standard form (TP import only for rad)
                      '#define NATOMS               23\n',  # Number of atoms for chemical equil
                      '#define NMOLECULES           172\n',  # Number of molecules for chemical equil
                      '#define MOL_DATA_FILE        "Data/molecules_all.dat"\n',  # Data file for chemical equilibrium calculation
                      # '#define ATOM_ABUN_FILE       "Data/atom_H2O_CH4.dat"\n',  # Data file for chemical equilibrium calculation
                      '#define IMPORTFILEX          "Result/Aux/Conx.dat"\n',  # File of concentrations X to be imported
                      '#define IMPORTFILEF          "Result/Aux/Conf.dat"\n',  # File of concentrations F to be imported
                      # '#define IFIMPORTH2O          0\n',  # When H2O is set to constant, 1=import mixing ratios
                      # '#define IFIMPORTCO2          0\n',  # When CO is set to constant, 1=import mixing ratios
                      # Reaction Zones
                      '#define REACTION_LIST        "Data/zone_Earth_Full.dat"\n',
                      '#define NKin                 645\n',  # Number of Regular Chemical Reaction in the standard list
                      '#define NKinM                90\n',  # Number of Thermolecular Reaction in the standard list
                      '#define NKinT                93\n',  # Number of Thermal Dissociation Reaction in the standard list
                      '#define NPho                 71\n',  # Number of Photochemical Reaction in the standard list
                      '#define THREEBODY            1.0\n',  # Enhancement of THREEBODY Reaction when CO2 dominant
                      #
                      # Parametization of Eddy Diffusion Coefficient
                      '#define EDDYPARA             1\n',  # =1 from Parametization, =2 from imported list
                      '#define KET                  1.0E+6\n',  # unit cm2 s-1
                      '#define KEH                  1.0E+6\n',
                      '#define ZT                   200.0\n',  # unit km
                      '#define Tback                1E+4\n',
                      '#define KET1                 1.0E+6\n',
                      '#define KEH1                 1.0E+8\n',
                      '#define EDDYIMPORT           "Data/EddyH2.dat"\n',
                      '#define MDIFF_H_1            4.87\n',
                      '#define MDIFF_H_2            0.698\n',
                      '#define MDIFF_H2_1           2.80\n',
                      '#define MDIFF_H2_2           0.740\n',
                      '#define MDIFF_H2_F           1.0\n',
                      #
                      # Parameters of rainout rates
                      '#define RainF                0.0\n',  # Rainout factor, 0 for no rainout, 1 for earthlike normal rainout, <1 for reduced rainout
                      '#define CloudDen             1.0\n',  # Cloud density in the unit of g m-3
                      # Output Options
                      '#define OUT_DIR              "Result/Retrieval_' + str(self.process) + '/"\n',
                      '#define TINTSET              20.0\n',  # Internal Heat Temperature
                      '\n',
                      '#define OUT_STD              "Result/Jupiter_1/ConcentrationSTD.dat"\n',
                      '#define OUT_FILE1            "Result/GJ1214_Figure/Conx.dat"\n',
                      '#define OUT_FILE2            "Result/GJ1214_Figure/Conf.dat"\n',
                      '#define NPRINT               1E+2\n',  # Printout results and histories every NPRINT iterations
                      '#define HISTORYPRINT         0\n',  # print out time series of chemical composition if set to 1
                      #
                      # Input choices for the infrared opacities
                      # Must be set to the same as the opacity code
                      #
                      '#define CROSSHEADING         "Cross3/N2_FullT_LowRes/"\n',
                      #
                      '#define NTEMP                20\n',  # Number of temperature points in grid
                      '#define TLOW                 100.0\n',  # Temperature range in K
                      '#define THIGH                2000.0\n',
                      #
                      '#define NPRESSURE            10\n',  # Number of pressure points in grid
                      '#define PLOW                 1.0e-01\n',  # Pressure range in Pa
                      '#define PHIGH                1.0e+08\n',
                      #
                      '#define NLAMBDA              16000\n',  # Number of wavelength points in grid
                      '#define LAMBDALOW            1.0e-07\n',  # Wavelength range in m -> 0.1 micron
                      '#define LAMBDAHIGH           2.0e-04\n',  # in m -> 200 micron
                      '#define LAMBDATYPE           1\n',  # LAMBDATYPE=1 -> constant resolution
                      #                                    # LAMBDATYPE=2 -> constant wave step
                      #
                      #
                      # IR emission spectra output options
                      '#define IRLamMin             1.0\n',  # Minimum wavelength in the IR emission output, in microns
                      '#define IRLamMax             100.0\n',  # Maximum wavelength in the IR emission output, in microns, was 100
                      '#define IRLamBin             9999\n',  # Number of wavelength bin in the IR emission spectra, was 9999
                      '#define Var1STD              7\n',
                      '#define Var2STD              20\n',
                      '#define Var3STD              21\n',
                      '#define Var4STD              52\n',
                      '#define Var1RATIO            0.0\n',
                      '#define Var2RATIO            0.0\n',
                      '#define Var3RATIO            0.0\n',
                      '#define Var4RATIO            0.0\n',
                      #
                      #  Stellar Light Reflection output options
                      '#define UVRFILE              "Result/Jupiter_1/Reflection"\n',  # Output spectrum file name
                      '#define UVRFILEVar1          "Result/Jupiter_1/ReflectionVar1.dat"\n',  # Output spectrum file name
                      '#define UVRFILEVar2          "Result/Jupiter_1/ReflectionVar2.dat"\n',  # Output spectrum file name
                      '#define UVRFILEVar3          "Result/Jupiter_1/ReflectionVar3.dat"\n',  # Output spectrum file name
                      '#define UVRFILEVar4          "Result/Jupiter_1/ReflectionVar4.dat"\n',  # Output spectrum file name
                      '#define UVROPTFILE           "Result/Jupiter_1/UVROpt.dat"\n',  # Output spectrum file name
                      '#define AGFILE               "Result/Jupiter_1/GeometricA.dat"\n',  # Output spectrum file name
                      #
                      # Stellar Light Transmission output options
                      '#define UVTFILE              "Result/Jupiter_1/Transmission.dat"\n',  # Output spectrum file name
                      '#define UVTFILEVar1          "Result/Jupiter_1/TransmissionVar1.dat"\n',  # Output spectrum file name
                      '#define UVTFILEVar2          "Result/Jupiter_1/TransmissionVar2.dat"\n',  # Output spectrum file name
                      '#define UVTFILEVar3          "Result/Jupiter_1/TransmissionVar3.dat"\n',  # Output spectrum file name
                      '#define UVTFILEVar4          "Result/Jupiter_1/TransmissionVar4.dat"\n',  # Output spectrum file name
                      '#define UVTOPTFILE           "Result/Jupiter_1/UVTOpt.dat"\n',  # Output spectrum file name
                      #
                      # Thermal Emission output options
                      '#define IRFILE               "Result/Jupiter_1/Emission.dat"\n',  # Output spectrum file name
                      '#define IRFILEVar1           "Result/Jupiter_1/EmissionVar1.dat"\n',  # Output spectrum file name
                      '#define IRFILEVar2           "Result/Jupiter_1/EmissionVar2.dat"\n',  # Output spectrum file name
                      '#define IRFILEVar3           "Result/Jupiter_1/EmissionVar3.dat"\n',  # Output spectrum file name
                      '#define IRFILEVar4           "Result/Jupiter_1/EmissionVar4.dat"\n',  # Output spectrum file name
                      '#define IRCLOUDFILE          "Result/Jupiter_1/CloudTopE.dat"\n',  # Output emission cloud top file name
                      #
                      # Cloud Top Determination
                      '#define OptCloudTop          1.0\n',  # Optical Depth of the Cloud Top
                      #
                      '#endif\n',
                      #
                      # 1 Tg yr-1 = 3.7257E+9 H /cm2/s for earth
                      ]
        with open(self.c_code_directory + 'par_' + str(self.process) + '.h', 'w') as file:
            for riga in c_par_file:
                file.write(riga)

    def __core_c_file(self):
        # if self.param['spectrum']['wl'][0] <= 0.4 and 0.75 < self.param['spectrum']['wl'][-1] < 1.1:
        #     iniz, fine = 1350, 4900
        # elif self.param['spectrum']['wl'][0] >= 0.4 and 0.75 < self.param['spectrum']['wl'][-1] < 1.1:
        #     iniz, fine = 3000, 4900
        # elif 0.4 <= self.param['spectrum']['wl'][0] < 0.9 and self.param['spectrum']['wl'][-1] > 1.1:
        #     iniz, fine = 3000, 6150
        # elif self.param['spectrum']['wl'][0] >= 0.9:
        #     iniz, fine = 4800, 6150
        # else:
        #     iniz, fine = 1350, 6150
        iniz = self.param['start_c_wl_grid'] + 0.0
        fine = self.param['stop_c_wl_grid'] + 0.0

        c_core_file = ['#include <stdio.h>\n',
                       '#include <math.h>\n',
                       '#include <stdlib.h>\n',
                       '#include <string.h>\n',

                       '#include "par_' + str(self.process) + '.h"\n',

                       '#include "constant.h"\n',
                       '#include "routine.h"\n',
                       '#include "global_rad_gasplanet.h"\n',
                       '#include "GetData.c"\n',
                       '#include "Interpolation.c"\n',
                       '#include "nrutil.h"\n',
                       '#include "nrutil.c"\n',
                       '#include "Convert.c"\n',
                       '#include "TPPara.c"\n',
                       # #include "TPScale.c"\n',
                       '#include "RefIdx.c"\n',
                       '#include "readcross.c"\n',
                       '#include "readcia.c"\n',
                       '#include "Reflection_.c"\n',
                       '#include "Trapz.c"\n',

                       # external (global) variables

                       'double thickl[zbin];\n',
                       'double zl[zbin+1];\n',
                       'double pl[zbin+1];\n',
                       'double tl[zbin+1];\n',
                       'double MM[zbin+1];\n',
                       'double MMZ[zbin+1];\n',
                       'double wavelength[NLAMBDA];\n',
                       'double solar[NLAMBDA];\n',
                       'double PSURFAB[NLAMBDA];\n',
                       'double crossr[zbin+1][NLAMBDA], crossr_H2O[NLAMBDA], crossr_CH4[NLAMBDA], crossr_CO2[NLAMBDA], crossr_CO[NLAMBDA], crossr_O2[NLAMBDA], crossr_O3[NLAMBDA], crossr_N2O[NLAMBDA], crossr_N2[NLAMBDA], crossr_H2[NLAMBDA];\n',
                       'double crossa[3][NLAMBDA], sinab[3][NLAMBDA], asym[3][NLAMBDA];\n',
                       'double **opacH2O, **opacNH3, **opacCH4, **opacH2S, **opacSO2, **opacCO2, **opacCO, **opacO2, **opacO3, **opacN2O, **opacN2;\n',

                       #double **opacH2O2, **opacHO2; \n',
                       #double **opacC2H2, **opacC2H4, **opacC2H6, **opacHCN, **opacCH2O2, **opacHNO3;\n',
                       #double **opacNO, **opacNO2, **opacOCS;\n',
                       #double **opacHF, **opacHCl, **opacHBr, **opacHI, **opacClO, **opacHClO;\n',
                       #double **opacHBrO, **opacPH3, **opacCH3Cl, **opacCH3Br, **opacDMS, **opacCS2;\n',

                       'int    ReactionR[NKin+1][7], ReactionM[NKinM+1][5], ReactionP[NPho+1][9], ReactionT[NKinT+1][4];\n',
                       'int    numr=0, numm=0, numt=0, nump=0, numx=0, numc=0, numf=0, numa=0, waternum=0, waterx=0;\n',
                       'double **xx, **xx1, **xx2, **xx3, **xx4;\n',
                       'double TransOptD[zbin+1][NLAMBDA], RefOptD[zbin+1][NLAMBDA];\n',
                       # /*double H2CIA[zbin+1][NLAMBDA], H2HeCIA[zbin+1][NLAMBDA], N2CIA[zbin+1][NLAMBDA], CO2CIA[zbin+1][NLAMBDA];*/\n',
                       'double H2H2CIA[zbin+1][NLAMBDA], H2HeCIA[zbin+1][NLAMBDA], H2HCIA[zbin+1][NLAMBDA], N2H2CIA[zbin+1][NLAMBDA], N2N2CIA[zbin+1][NLAMBDA], CO2CO2CIA[zbin+1][NLAMBDA], O2O2CIA[zbin+1][NLAMBDA];\n',
                       'double cH2O[zbin+1][NLAMBDA], aH2O[zbin+1][NLAMBDA], gH2O[zbin+1][NLAMBDA];\n',
                       'double cNH3[zbin+1][NLAMBDA], aNH3[zbin+1][NLAMBDA], gNH3[zbin+1][NLAMBDA];\n',

                       'int main()\n',
                       '{\n',
                       '    int s,i,ii,j,jj,jjj,k,nn,qytype,stdnum;\n',
                       '    int nums, numx1=1, numf1=1, numc1=1, numr1=1, numm1=1, nump1=1, numt1=1;\n',
                       '    char *temp;\n',
                       '    char dataline[10000];\n',
                       '    double temp1, wavetemp, crosstemp, DD, GA, mixtemp;\n',
                       '    double z[zbin+1], T[zbin+1], PP[zbin+1], P[zbin+1];\n',
                       '    double *wavep, *crossp, *crosspa, *qyp, *qyp1, *qyp2, *qyp3, *qyp4, *qyp5, *qyp6, *qyp7, **cross, **qy;\n',
                       '    double **crosst, **qyt, *crosspt, *qypt, *qyp1t, *qyp2t, *qyp3t, *qyp4t, *qyp5t, *qyp6t, *qyp7t;\n',
                       '    FILE *fspecies, *fzone, *fhenry, *fp, *fp1, *fp2, *fp3;\n',
                       '    FILE *fout, *fout1, *fout3, *fout4, *fcheck, *ftemp, *fout5, *foutp, *foutc;\n',
                       '    FILE *fimport, *fimportcheck;\n',
                       '    FILE *TPPrint;\n',

                       '    xx = dmatrix(1,zbin,1,NSP);\n',
                       '    xx1 = dmatrix(1,zbin,1,NSP);\n',
                       '    xx2 = dmatrix(1,zbin,1,NSP);\n',
                       '    xx3 = dmatrix(1,zbin,1,NSP);\n',
                       '    xx4 = dmatrix(1,zbin,1,NSP);\n',

                       '    for (j=1; j<=zbin; j++) {\n',
                       '        for (i=1; i<=NSP; i++) {\n',
                       '            xx[j][i] = 0.0;\n',
                       '            xx1[j][i] = 0.0;\n',
                       '            xx2[j][i] = 0.0;\n',
                       '            xx3[j][i] = 0.0;\n',
                       '            xx4[j][i] = 0.0;\n',
                       '        }\n',
                       '    }\n',

                       #    GA = GRAVITY*MASS_PLANET/RADIUS_PLANET/RADIUS_PLANET;\n',  # Planet Surface Gravity Acceleration, in SI

                       #    Set the wavelength for calculation
                       '    double dlambda, start, interval, lam[NLAMBDA];\n',
                       '    start = log10(LAMBDALOW);\n',
                       '    interval = log10(LAMBDAHIGH) - log10(LAMBDALOW);\n',
                       '    dlambda = interval / (NLAMBDA-1.0);\n',
                       '    for (i=0; i<NLAMBDA; i++){\n',
                       '        wavelength[i] = pow(10.0, start+i*dlambda)*1.0E+9;\n',  # in nm
                       '        lam[i] = wavelength[i]*1.0E-3;\n',  # in microns
                       '    }\n',

                       # Obtain the stellar radiation
                       '    fp2 = fopen(STAR_SPEC,"r");\n',
                       '    fp3 = fopen(STAR_SPEC,"r");\n',
                       '    s = LineNumber(fp2, 1000);\n',
                       '    double swave[s], sflux[s];\n',
                       '    GetData(fp3, 1000, s, swave, sflux);\n',
                       '    fclose(fp2);\n',
                       '    fclose(fp3);\n',
                       '    Interpolation(wavelength, NLAMBDA, solar, swave, sflux, s, 0);\n',
                       '    for (i=0; i<NLAMBDA; i++) {\n',
                       '        solar[i] = solar[i]/ORBIT/ORBIT*FaintSun;\n',  # convert from flux at 1 AU
                       '    }\n',
                       '    i=0;\n',
                       '    while (solar[i]>0 || wavelength[i]<9990 ) { i++;}\n',
                       '    for (j=i; j<NLAMBDA; j++) {\n',
                       '        solar[j] = solar[i-1]*pow(wavelength[i-1],4)/pow(wavelength[j],4);\n',
                       '    }\n',
                       # '\t  printf("%s\\n", "The stellar radiation data are imported.");\n',
                       '    fp2 = fopen(SURF_SPEC,"r");\n',
                       '    char dataline2[100];\n',
                       '    i=0;\n',
                       '    while (i < NLAMBDA && fgets(dataline2, sizeof(dataline2), fp2) != NULL) {\n',
                       '        PSURFAB[i] = atof(dataline2);\n',  # convert string to double and store in array
                       '        i++;\n',
                       '    }\n',
                       '    fclose(fp2);\n',
                       # Import Species List
                       '    fspecies=fopen(SPECIES_LIST, "r");\n',
                       '    s=LineNumber(fspecies, 10000);\n',
                       # '\tprintf("Species list: \\n");\n',
                       '    fclose(fspecies);\n',
                       '    fspecies=fopen(SPECIES_LIST, "r");\n',
                       '    struct Molecule species[s];\n',
                       '    temp=fgets(dataline, 10000, fspecies);\n',  # Read in the header line
                       '    i=0;\n',
                       '    while (fgets(dataline, 10000, fspecies) != NULL )\n',
                       '    {\n',
                       '        sscanf(dataline, "%s %s %d %d %lf %lf %d %lf %lf", (species+i)->name, (species+i)->type, &((species+i)->num), &((species+i)->mass), &((species+i)->mix), &((species+i)->upper), &((species+i)->lowertype), &((species+i)->lower), &((species+i)->lower1));\n',
                       # '\t\tprintf("%s %s %d %d %lf %lf %d %lf %lf\\n",(species+i)->name, (species+i)->type, (species+i)->num, (species+i)->mass, (species+i)->mix, (species+i)->upper, (species+i)->lowertype, (species+i)->lower, (species+i)->lower1);\n',
                       '        if (strcmp("X",species[i].type)==0) {numx=numx+1;}\n',
                       '        if (strcmp("F",species[i].type)==0) {numf=numf+1;}\n',
                       '        if (strcmp("C",species[i].type)==0) {numc=numc+1;}\n',
                       '        if (strcmp("A",species[i].type)==0) {numx=numx+1; numa=numa+1;}\n',
                       '        i=i+1;\n',
                       '    }\n',
                       '    fclose(fspecies);\n',
                       '    nums=numx+numf+numc;\n',
                       # '\tprintf("%s\\n", "The species list is imported.");\n',
                       # '\tprintf("%s %d\\n", "Number of species in model:", nums);\n',
                       # '\tprintf("%s %d\\n", "Number of species to be solved in full:", numx);\n',
                       # '\tprintf("%s %d\\n", "In which the number of aerosol species is:", numa);\n',
                       # '\tprintf("%s %d\\n", "Number of species to be solved in photochemical equil:", numf);\n',
                       # '\tprintf("%s %d\\n", "Number of species assumed to be constant:", numc);\n',
                       '    int labelx[numx+1], labelc[numc+1], labelf[numf+1], MoleculeM[numx+1], listAER[numa+1], AERCount=1;\n',
                       '    for (i=0; i<s; i++) {\t\t\t\n',
                       '        if (strcmp("X",species[i].type)==0 || strcmp("A",species[i].type)==0) {\n',
                       '            labelx[numx1]=species[i].num;\n',
                       '            for (j=1; j<=zbin; j++) { \n',
                       '                xx[j][species[i].num]=MM[j]*species[i].mix;\n',
                       '            }\n',
                       '            if (species[i].num==7) {\n',
                       '                waternum=numx1;\n',
                       '                waterx=1;\n',
                       '            }\n',
                       '            MoleculeM[numx1]=species[i].mass;\n',
                       '            if (species[i].lowertype==1) {\n',
                       '                xx[1][species[i].num]=species[i].lower1*MM[1];\n',
                       '            }\n',
                       '            if (strcmp("A",species[i].type)==0) {\n',
                       '                listAER[AERCount]=numx1;\n',
                       '                AERCount = AERCount+1;\n',
                       #                printf("%s %d\\n", "The aerosol species is", numx1);\n',
                       '            }\n',
                       '            numx1=numx1+1;\n',
                       '        }\n',
                       '        if (strcmp("F",species[i].type)==0) {\n',
                       '            labelf[numf1]=species[i].num;\n',
                       '            for (j=1; j<=zbin; j++) { \n',
                       '                xx[j][species[i].num]=MM[j]*species[i].mix;\n',
                       '            }\n',
                       '            numf1=numf1+1;\n',
                       '        }\n',
                       '        if (strcmp("C",species[i].type)==0) {\n',
                       '            labelc[numc1]=species[i].num;\n',
                       '            for (j=1; j<=zbin; j++) {\n',
                       '                xx[j][species[i].num]=MM[j]*species[i].mix;\n',
                       '            }\n',
                       # import constant mixing ratio list for H2O
                       #            if (IFIMPORTH2O == 1 && species[i].num == 7) {\n',
                       #                fimport=fopen("Data/ConstantMixing.dat", "r");\n',
                       #                fimportcheck=fopen("Data/ConstantMixingH2O.dat", "w");\n',
                       #                temp=fgets(dataline, 10000, fimport);\n',  # Read in the header line
                       #                for (j=1; j<=zbin; j++) {\n',
                       #                    fscanf(fimport, "%lf\\t", &temp1);\n',
                       #                    fscanf(fimport, "%le\\t", &mixtemp);\n',
                       #                    fscanf(fimport, "%le\\t", &temp1);\n',
                       #                    xx[j][7]=mixtemp * MM[j];\n',
                       #                    fprintf(fimportcheck, "%f\\t%e\\t%e\\n", zl[j], mixtemp, xx[j][7]);\n',
                       #                }\n',
                       #                fclose(fimport);\n',
                       #                fclose(fimportcheck);\n',
                       #            }\n',
                       # import constant mixing ratio list for CO2
                       #            if (IFIMPORTCO2 == 1 && species[i].num == 52) {\n',
                       #                fimport=fopen("Data/ConstantMixing.dat", "r");\n',
                       #                fimportcheck=fopen("Data/ConstantMixingCO2.dat", "w");\n',
                       #                temp=fgets(dataline, 10000, fimport);\n',  # Read in the header line
                       #                for (j=1; j<=zbin; j++) {\n',
                       #                    fscanf(fimport, "%lf\\t", &temp1);\n',
                       #                    fscanf(fimport, "%le\\t", &temp1);\n',
                       #                    fscanf(fimport, "%le\\t", &mixtemp);\n',
                       #                    xx[j][52]=mixtemp * MM[j];\n',
                       #                    fprintf(fimportcheck, "%f\\t%e\\t%e\\n", zl[j], mixtemp, xx[j][52]);\n',
                       #                }\n',
                       #                fclose(fimport);\n',
                       #                fclose(fimportcheck);\n',
                       #            }\n',
                       '            numc1=numc1+1;\n',
                       '        }\n',
                       '    }\n',
                       '    fimport=fopen(IMPORTFILEX, "r");\n',
                       #    fimportcheck=fopen("Data/Fimportcheck.dat","w");\n',
                       '    temp=fgets(dataline, 10000, fimport);\n',  # Read in the header line
                       '    temp=fgets(dataline, 10000, fimport);\n',  # Read in the header line
                       '    for (j=1; j<=zbin; j++) {\n',
                       '        fscanf(fimport, "%lf\\t\\t", &temp1);\n',
                       #        fprintf(fimportcheck, "%lf\\t", temp1);\n',
                       '        for (i=1; i<=numx; i++) {\n',
                       '            fscanf(fimport, "%le\\t\\t", &xx[j][labelx[i]]);\n',
                       #            fprintf(fimportcheck, "%e\\t", xx[j][labelx[i]]);\n',
                       '        }\n',
                       '        fscanf(fimport, "%lf\\t\\t", &temp1);\n',  # column of air
                       #        fprintf(fimportcheck,"\\n");\n',
                       '    }\n',
                       '    fclose(fimport);\n',
                       '    fimport=fopen(IMPORTFILEF, "r");\n',
                       '    temp=fgets(dataline, 10000, fimport);\n',  # Read in the header line
                       '    temp=fgets(dataline, 10000, fimport);\n',  # Read in the header line
                       '    for (j=1; j<=zbin; j++) {\n',
                       '        fscanf(fimport, "%lf\\t\\t", &temp1);\n',
                       #        fprintf(fimportcheck, "%lf\\t", temp1);\n',
                       '        for (i=1; i<=numf; i++) {\n',
                       '            fscanf(fimport, "%le\\t\\t", &xx[j][labelf[i]]);\n',
                       #            fprintf(fimportcheck, "%e\\t", xx[j][labelf[i]]);\n',
                       '        }\n',
                       '        fscanf(fimport, "%lf\\t\\t", &temp1);\n',  # column of air
                       #        fprintf(fimportcheck,"\\n");\n',
                       '    }\n',
                       '    fclose(fimport);\n',
                       #    fclose(fimportcheck);\n',

                       # Set up atmospheric profiles

                       '    char outstd[1024];\n',
                       '    strcpy(outstd,OUT_DIR);\n',
                       '    strcat(outstd,"ConcentrationSTD.dat");\n',
                       '    if (IMODE == 4) {\n',  # Import the computed profile directly
                       '        fimport=fopen(outstd, "r");\n',
                       #        fimportcheck=fopen("Data/Fimportcheck.dat","w");\n',
                       '        temp=fgets(dataline, 10000, fimport);\n',  # Read in the header line
                       '        temp=fgets(dataline, 10000, fimport);\n',  # Read in the header line
                       '        for (j=1; j<=zbin; j++) {\n',
                       '            fscanf(fimport, "%lf\\t", &zl[j]);\n',
                       #            fprintf(fimportcheck, "%lf\\t", zl[j]);\n',
                       '            fscanf(fimport, "%lf\\t", &z[j-1]);\n',
                       '            fscanf(fimport, "%lf\\t", &z[j]);\n',
                       '            fscanf(fimport, "%lf\\t", &tl[j]);\n',
                       '            fscanf(fimport, "%le\\t", &pl[j]);\n',
                       '            MM[j] = pl[j]/KBOLTZMANN/tl[j]*1.0E-6;\n',
                       '            for (i=1; i<=NSP; i++) {\n',
                       '                fscanf(fimport, "%le\\t", &xx[j][i]);\n',
                       #                fprintf(fimportcheck, "%e\\t", xx[j][i]);\n',
                       #                MM[j] += xx[j][i];\n',
                       '            }\n',
                       #            printf("%s %f %f\\n", "TP", tl[j], pl[j]);\n',
                       #            fprintf(fimportcheck,"\\n");\n',
                       '        }\n',
                       '        fclose(fimport);\n',
                       #        fclose(fimportcheck);\n',
                       # '        thickl = (z[zbin]-z[zbin-1])*1.0E+5;\n',
                       '        for (j=1; j<=zbin; j++) {\n',
                       '            thickl[j] = (z[j]-z[j-1])*1.0E+5;\n',
                       # '            printf("%f\\n", thickl[j]);\n',
                       '        }\n',
                       '        for (j=1; j<zbin; j++) {\n',
                       '            T[j] = (tl[j] + tl[j+1])/2.0;\n',
                       '        }\n',
                       '        T[0] = 1.5*tl[1] - 0.5*tl[2];\n',
                       '        T[zbin] = 1.5*tl[zbin] - 0.5*tl[zbin-1];\n',
                       '    }\n',

                       #    Rayleigh Scattering
                       '    double refidx0,DenS;\n',
                       '    DenS = 101325.0 / KBOLTZMANN / 273.0 * 1.0E-6;\n',
                       # '    for (i=0; i<NLAMBDA; i++){\n',
                       '    for (i=' + str(iniz) + '; i<' + str(fine) + '; i++){\n',
                       #        if (RefIdxType == 0) { refidx0=AirRefIdx(wavelength[i]);}\n',
                       #        if (RefIdxType == 1) { refidx0=CO2RefIdx(wavelength[i]);}\n',
                       #        if (RefIdxType == 2) { refidx0=HeRefIdx(wavelength[i]);}\n',
                       #        if (RefIdxType == 3) { refidx0=N2RefIdx(wavelength[i]);}\n',
                       #        if (RefIdxType == 4) { refidx0=NH3RefIdx(wavelength[i]);}\n',
                       #        if (RefIdxType == 5) { refidx0=CH4RefIdx(wavelength[i]);}\n',
                       #        if (RefIdxType == 6) { refidx0=H2RefIdx(wavelength[i]);}\n',
                       #        if (RefIdxType == 7) { refidx0=O2RefIdx(wavelength[i]);}\n',
                       #        if (refidx0 < 1.0) { refidx0 = 1.0; }\n',
                       #        crossr[i]=1.061*8.0*pow(PI,3)*pow(pow(refidx0,2)-1,2)/3.0/pow(wavelength[i]*1.0E-7,4)/DenS/DenS;\n',
                       #        if (RefIdxType == 6) {crossr[i] = 8.14e-13*pow(wavelength[i]*10.0,-4)+1.28e-6*pow(wavelength[i]*10.0,-6)+1.61*pow(wavelength[i]*10.0,-8); }\n',  # Dalgarno 1962
                       #        if (RefIdxType == 8) {\n',
                       '        refidx0 = H2ORefIdx(wavelength[i]);\n',
                       '        if (refidx0 < 1.0) { refidx0 = 1.0; }\n',
                       '        crossr_H2O[i] = 1.061*8.0*pow(PI,3)*pow(pow(refidx0,2)-1,2)/3.0/pow(wavelength[i]*1.0E-7,4)/DenS/DenS;\n',
                       '        refidx0 = CH4RefIdx(wavelength[i]);\n',
                       '        if (refidx0 < 1.0) { refidx0 = 1.0; }\n',
                       '        crossr_CH4[i] = 1.061*8.0*pow(PI,3)*pow(pow(refidx0,2)-1,2)/3.0/pow(wavelength[i]*1.0E-7,4)/DenS/DenS;\n',
                       '        refidx0 = CO2RefIdx(wavelength[i]);\n',
                       '        if (refidx0 < 1.0) { refidx0 = 1.0; }\n',
                       '        crossr_CO2[i] = 1.061*8.0*pow(PI,3)*pow(pow(refidx0,2)-1,2)/3.0/pow(wavelength[i]*1.0E-7,4)/DenS/DenS;\n',
                       '        refidx0 = CORefIdx(wavelength[i]);\n',
                       '        if (refidx0 < 1.0) { refidx0 = 1.0; }\n',
                       '        crossr_CO[i] = 1.061*8.0*pow(PI,3)*pow(pow(refidx0,2)-1,2)/3.0/pow(wavelength[i]*1.0E-7,4)/DenS/DenS;\n',
                       '        refidx0 = O2RefIdx(wavelength[i]);\n',
                       '        if (refidx0 < 1.0) { refidx0 = 1.0; }\n',
                       '        crossr_O2[i] = 1.061*8.0*pow(PI,3)*pow(pow(refidx0,2)-1,2)/3.0/pow(wavelength[i]*1.0E-7,4)/DenS/DenS;\n',
                       '        refidx0 = O3RefIdx(wavelength[i]);\n',
                       '        if (refidx0 < 1.0) { refidx0 = 1.0; }\n',
                       '        crossr_O3[i] = 1.061*8.0*pow(PI,3)*pow(pow(refidx0,2)-1,2)/3.0/pow(wavelength[i]*1.0E-7,4)/DenS/DenS;\n',
                       '        refidx0 = N2ORefIdx(wavelength[i]);\n',
                       '        if (refidx0 < 1.0) { refidx0 = 1.0; }\n',
                       '        crossr_N2O[i] = 1.061*8.0*pow(PI,3)*pow(pow(refidx0,2)-1,2)/3.0/pow(wavelength[i]*1.0E-7,4)/DenS/DenS;\n',
                       '        refidx0 = N2RefIdx(wavelength[i]);\n',
                       '        if (refidx0 < 1.0) { refidx0 = 1.0; }\n',
                       '        crossr_N2[i] = 1.061*8.0*pow(PI,3)*pow(pow(refidx0,2)-1,2)/3.0/pow(wavelength[i]*1.0E-7,4)/DenS/DenS;\n',
                       '        crossr_H2[i] = 8.14e-13*pow(wavelength[i]*10.0,-4)+1.28e-6*pow(wavelength[i]*10.0,-6)+1.61*pow(wavelength[i]*10.0,-8);\n',  # Dalgarno 1962
                       '        for (j=1; j<=zbin; j++) {\n']
        if not self.param['contribution']:
            c_core_file += ['            crossr[j][i] = ((crossr_H2O[i] * (xx[j][7]/MM[j])) + (crossr_CH4[i] * (xx[j][21]/MM[j])) + (crossr_CO2[i] * (xx[j][52]/MM[j])) + (crossr_CO[i] * (xx[j][20]/MM[j])) + (crossr_O2[i] * (xx[j][54]/MM[j])) + (crossr_O3[i] * (xx[j][2]/MM[j])) + (crossr_N2O[i] * (xx[j][11]/MM[j])) + (crossr_N2[i] * (xx[j][55]/MM[j])) + (crossr_H2[i] * (xx[j][53]/MM[j]))) / ((xx[j][7]/MM[j]) + (xx[j][21]/MM[j]) + (xx[j][52]/MM[j]) + (xx[j][20]/MM[j]) + (xx[j][54]/MM[j]) + (xx[j][2]/MM[j]) + (xx[j][11]/MM[j]) + (xx[j][55]/MM[j]) + (xx[j][53]/MM[j]));\n']
        elif self.param['contribution'] and self.param['mol_contr'] is not None:
            c_core_file += ['            crossr[j][i] = ((crossr_H2O[i] * (xx[j][7]/MM[j])) + (crossr_CH4[i] * (xx[j][21]/MM[j])) + (crossr_CO2[i] * (xx[j][52]/MM[j])) + (crossr_CO[i] * (xx[j][20]/MM[j])) + (crossr_O2[i] * (xx[j][54]/MM[j])) + (crossr_O3[i] * (xx[j][2]/MM[j])) + (crossr_N2O[i] * (xx[j][11]/MM[j])) + (crossr_N2[i] * (xx[j][55]/MM[j])) + (crossr_H2[i] * (xx[j][53]/MM[j])));\n']
        else:
            c_core_file += ['            crossr[j][i] = 0.0;\n']
        c_core_file+= ['        }\n',
                       '    }\n',

                       '    readcia();\n',

                       # check CIA
                       #    for (i=0; i<NLAMBDA; i++) {\n',
                       #    printf("%s\\t%f\\t%e\\t%e\\t%e\\t%e\\n", "CIA", wavelength[i], H2CIA[1][i], H2HeCIA[1][i], N2CIA[1][i], CO2CIA[1][i]);\n',
                       #    }\n',
                       #
                       # '\tprintf("%s\\n", "Collision-induced absorption cross sections are imported ");\n',

                       # Obtain the opacity
                       '    opacH2O = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacNH3 = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacCH4 = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacH2S = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacSO2 = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacCO2 = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacCO = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacO2 = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacO3 = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacN2O = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacN2 = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacOH = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacH2CO = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacH2O2 = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacHO2 = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacC2H2 = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacC2H4 = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacC2H6 = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacHCN = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacCH2O2 = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacHNO3 = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacN2O = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacNO = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacNO2 = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacOCS = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacHF = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacHCl = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacHBr = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacHI = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacClO = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacHClO = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacHBrO = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacPH3 = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacCH3Cl = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacCH3Br = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacDMS = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacCS2 = dmatrix(1,zbin,0,NLAMBDA-1);\n',

                       '    char crossfile[1024];\n']

        for mol in self.param['fit_molecules']:
            if mol != 'H2':
                c_core_file += ['    strcpy(crossfile,CROSSHEADING);\n',
                                '    strcat(crossfile,"opac' + mol + '.dat");\n',
                                '    readcross(crossfile, opac' + mol + ');\n']

                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacCO.dat");\n',
                       #    readcross(crossfile, opacCO);\n',
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacN2O.dat");\n',
                       #    readcross(crossfile, opacN2O);\n',
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacOH.dat");\n',
                       #    readcross(crossfile, opacOH);\n',
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacH2CO.dat");\n',
                       #    readcross(crossfile, opacH2CO);\n',
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacH2O2.dat");\n',
                       #    readcross(crossfile, opacH2O2);\n',
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacHO2.dat");\n',
                       #    readcross(crossfile, opacHO2);\n',
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacC2H2.dat");\n',
                       #    readcross(crossfile, opacC2H2);\n',
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacC2H4.dat");\n',
                       #    readcross(crossfile, opacC2H4);\n',
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacC2H6.dat");\n',
                       #    readcross(crossfile, opacC2H6);\n',
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacHCN.dat");\n',
                       #    readcross(crossfile, opacHCN);\n',
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacCH2O2.dat");\n',
                       #    readcross(crossfile, opacCH2O2);\n',
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacHNO3.dat");\n',
                       #    readcross(crossfile, opacHNO3);\n',
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacN2.dat");\n',
                       #    readcross(crossfile, opacN2);\n',
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacNO.dat");\n',
                       #    readcross(crossfile, opacNO);\n',
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacNO2.dat");\n',
                       #    readcross(crossfile, opacNO2);\n',
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacOCS.dat");\n',
                       #    readcross(crossfile, opacOCS);\n',
                       #
                       #    foutc = fopen("Data/IRCross.dat","w");\n',
                       #    for (i=0; i<NLAMBDA; i++) {\n',
                       #        fprintf(foutc, "%f\\t", wavelength[i]);\n',
                       #        fprintf(foutc, "%e\\t", opacCO2[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacO2[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacH2O[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacCH4[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacNH3[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacCO[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacO3[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacN2O[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacSO2[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacOH[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacH2CO[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacH2O2[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacHO2[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacH2S[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacC2H2[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacC2H4[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacC2H6[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacHCN[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacCH2O2[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacHNO3[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacN2[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacNO[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacNO2[1][i]);\n',
                       #        fprintf(foutc, "%e\\n", opacOCS[1][i]);\n',
                       #        fprintf(foutc, "%e\\n", opacHF[1][i]);\n',
                       #        fprintf(foutc, "%e\\n", opacHCl[1][i]);\n',
                       #        fprintf(foutc, "%e\\n", opacHBr[1][i]);\n',
                       #        fprintf(foutc, "%e\\n", opacHI[1][i]);\n',
                       #        fprintf(foutc, "%e\\n", opacClO[1][i]);\n',
                       #        fprintf(foutc, "%e\\n", opacHClO[1][i]);\n',
                       #        fprintf(foutc, "%e\\n", opacHBrO[1][i]);\n',
                       #        fprintf(foutc, "%e\\n", opacPH3[1][i]);\n',
                       #        fprintf(foutc, "%e\\n", opacCH3Cl[1][i]);\n',
                       #        fprintf(foutc, "%e\\n", opacCH3Br[1][i]);\n',
                       #        fprintf(foutc, "%e\\n", opacDMS[1][i]);\n',
                       #        fprintf(foutc, "%e\\n", opacCS2[1][i]);*/\n',
                       #    }\n',
                       #    fclose(foutc);\n',
                       #
                       #    tprintf("%s\\n", "Molecular cross sections are imported ");\n',

                       # Get Reaction List
        c_core_file+= ['    fzone=fopen(REACTION_LIST, "r");\n',
                       '    s=LineNumber(fzone, 10000);\n',
                       '    fclose(fzone);\n',
                       '    fzone=fopen(REACTION_LIST, "r");\n',
                       '    struct Reaction React[s];\n',
                       '    temp=fgets(dataline, 10000, fzone);\n',  # Read in the header line
                       '    i=0;\n',
                       '    while (fgets(dataline, 10000, fzone) != NULL )\n',
                       '    {\n',
                       '        sscanf(dataline, "%d %s %d", &((React+i)->dum), (React+i)->type, &((React+i)->num));\n',
                       #        printf("%d %s %d\\n", (React+i)->dum, React[i].type, React[i].num);\n',
                       '        if (strcmp("R",React[i].type)==0) {numr=numr+1;}\n',
                       '        if (strcmp("M",React[i].type)==0) {numm=numm+1;}\n',
                       '        if (strcmp("P",React[i].type)==0) {nump=nump+1;}\n',
                       '        if (strcmp("T",React[i].type)==0) {numt=numt+1;}\n',
                       '        i=i+1;\n',
                       '    }\n',
                       '    fclose(fzone);\n',
                       '    int zone_r[numr+1], zone_m[numm+1], zone_p[nump+1], zone_t[numt+1];\n',
                       '    for (i=0; i<s; i++) {\n',
                       '        if (strcmp("R",React[i].type)==0) {\n',
                       '            zone_r[numr1]=React[i].num;\n',
                       '            numr1=numr1+1;\n',
                       '        }\n',
                       '        if (strcmp("M",React[i].type)==0) {\n',
                       '            zone_m[numm1]=React[i].num;\n',
                       '            numm1=numm1+1;\n',
                       '        }\n',
                       '        if (strcmp("P",React[i].type)==0) {\n',
                       '            zone_p[nump1]=React[i].num;\n',
                       '            nump1=nump1+1;\n',
                       '        }\n',
                       '        if (strcmp("T",React[i].type)==0) {\n',
                       '            zone_t[numt1]=React[i].num;\n',
                       '        numt1=numt1+1;\n',
                       '        }\n',
                       '    }\n',
                       #    printf("%s\\n", "The reaction lists are imported.");\n',
                       #    tprintf("%s %d\\n", "Number of bi-molecular reactions:", numr);\n',
                       #    tprintf("%s %d\\n", "Number of tri-molecular reactions:", numm);\n',
                       #    tprintf("%s %d\\n", "Number of photolysis:", nump);\n',
                       #    tprintf("%s %d\\n", "Number of thermo-dissociations:", numt);\n',
                       '    GetReaction();\n',

                       # get the cross sections and quantum yields of molecules
                       '    cross=dmatrix(1,nump,0,NLAMBDA-1);\n',
                       '    crosst=dmatrix(1,nump,0,NLAMBDA-1);\n',
                       '    qy=dmatrix(1,nump,0,NLAMBDA-1);\n',
                       '    qyt=dmatrix(1,nump,0,NLAMBDA-1);\n',
                       '    int stdcross[nump+1];\n',
                       '    double qysum[nump+1];\n',
                       #    fcheck=fopen("Data/CrossSectionCheck.dat","w"); \n',
                       '    for (i=1; i<=nump; i++) {\n',
                       '        stdcross[i]=ReactionP[zone_p[i]][1];\n',
                       '        qytype=ReactionP[zone_p[i]][8];\n',
                       '        qysum[i]=ReactionP[zone_p[i]][7];\n',
                       '        j=0;\n',
                       '        while (species[j].num != stdcross[i]) {j=j+1;}\n',
                       #        printf("%s\\n",species[j].name);\n',
                       '        fp=fopen(species[j].name, "r");\n',
                       '        fp1=fopen(species[j].name, "r");\n',
                       '        s=LineNumber(fp, 1000);\n',
                       #        printf("%d\\n",s);\n',
                       '        wavep=dvector(0,s-1);\n',
                       '        crossp=dvector(0,s-1);\n',
                       '        qyp=dvector(0,s-1);\n',
                       '        qyp1=dvector(0,s-1);\n',
                       '        qyp2=dvector(0,s-1);\n',
                       '        qyp3=dvector(0,s-1);\n',
                       '        qyp4=dvector(0,s-1);\n',
                       '        qyp5=dvector(0,s-1);\n',
                       '        qyp6=dvector(0,s-1);\n',
                       '        qyp7=dvector(0,s-1);\n',
                       '        crosspt=dvector(0,s-1);\n',
                       '        qypt=dvector(0,s-1);\n',
                       '        qyp1t=dvector(0,s-1);\n',
                       '        qyp2t=dvector(0,s-1);\n',
                       '        qyp3t=dvector(0,s-1);\n',
                       '        qyp4t=dvector(0,s-1);\n',
                       '        qyp5t=dvector(0,s-1);\n',
                       '        qyp6t=dvector(0,s-1);\n',
                       '        qyp7t=dvector(0,s-1);\n',
                       '        k=0;\n',
                       '        if (qytype==1) {\n',
                       '            while (fgets(dataline, 1000, fp1) != NULL ) {\n',
                       '                sscanf(dataline, "%lf %le %le %lf %lf", wavep+k, crossp+k, crosspt+k, qyp+k, qypt+k);\n',
                       '                k=k+1; }\n',
                       '        }\n',
                       '        if (qytype==2) {\n',
                       '            while (fgets(dataline, 1000, fp1) != NULL ) {\n',
                       '                sscanf(dataline, "%lf %le %le %lf %lf %lf %lf", wavep+k, crossp+k, crosspt+k, qyp1+k, qyp1t+k, qyp+k, qypt+k);\n',
                       '                k=k+1; }\n',
                       '        }\n',
                       '        if (qytype==3) {\n',
                       '            while (fgets(dataline, 1000, fp1) != NULL ) {\n',
                       '                sscanf(dataline, "%lf %le %le %lf %lf %lf %lf %lf %lf", wavep+k, crossp+k, crosspt+k, qyp1+k, qyp1t+k, qyp2+k, qyp2t+k, qyp+k, qypt+k);\n',
                       '                k=k+1; }\n',
                       '        }\n',
                       '        if (qytype==4) {\n',
                       '            while (fgets(dataline, 1000, fp1) != NULL ) {\n',
                       '                sscanf(dataline, "%lf %le %le %lf %lf %lf %lf %lf %lf %lf %lf", wavep+k, crossp+k, crosspt+k, qyp1+k, qyp1t+k, qyp2+k, qyp2t+k, qyp3+k, qyp3t+k, qyp+k, qypt+k);\n',
                       '                k=k+1; }\n',
                       '        }\n',
                       '        if (qytype==5) {\n',
                       '            while (fgets(dataline, 1000, fp1) != NULL ) {\n',
                       '                sscanf(dataline, "%lf %le %le %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", wavep+k, crossp+k, crosspt+k, qyp1+k, qyp1t+k, qyp2+k, qyp2t+k, qyp3+k, qyp3t+k, qyp4+k, qyp4t+k, qyp+k, qypt+k);\n',
                       '                k=k+1; }\n',
                       '        }\n',
                       '        if (qytype==6) {\n',
                       '            while (fgets(dataline, 1000, fp1) != NULL ) {\n',
                       '                sscanf(dataline, "%lf %le %le %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", wavep+k, crossp+k, crosspt+k, qyp1+k, qyp1t+k, qyp2+k, qyp2t+k, qyp3+k, qyp3t+k, qyp4+k, qyp4t+k, qyp5+k, qyp5t+k, qyp+k, qypt+k);\n',
                       '                k=k+1; }\n',
                       '        }\n',
                       '        if (qytype==7) {\n',
                       '            while (fgets(dataline, 1000, fp1) != NULL ) {\n',
                       '                sscanf(dataline, "%lf %le %le %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", wavep+k, crossp+k, crosspt+k, qyp1+k, qyp1t+k, qyp2+k, qyp2t+k, qyp3+k, qyp3t+k, qyp4+k, qyp4t+k, qyp5+k, qyp5t+k, qyp6+k, qyp6t+k, qyp+k, qypt+k);\n',
                       '                k=k+1; }\n',
                       '        }\n',
                       '        if (qytype==8) {\n',
                       '            while (fgets(dataline, 1000, fp1) != NULL ) {\n',
                       '                sscanf(dataline, "%lf %le %le %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", wavep+k, crossp+k, crosspt+k, qyp1+k, qyp1t+k, qyp2+k, qyp2t+k, qyp3+k, qyp3t+k, qyp4+k, qyp4t+k, qyp5+k, qyp5t+k, qyp6+k, qyp6t+k, qyp7+k, qyp7t+k, qyp+k, qypt+k);\n',
                       '                k=k+1; }\n',
                       '        }\n',
                       '        fclose(fp);\n',
                       '        fclose(fp1);\n',
                       '        Interpolation(wavelength, NLAMBDA, *(cross+i), wavep, crossp, s, 0);\n',
                       '        Interpolation(wavelength, NLAMBDA, *(qy+i), wavep, qyp, s, 0);\n',
                       '        Interpolation(wavelength, NLAMBDA, *(crosst+i), wavep, crosspt, s, 0);\n',
                       '        Interpolation(wavelength, NLAMBDA, *(qyt+i), wavep, qypt, s, 0);\n',
                       '        free_dvector(wavep,0,s-1);\n',
                       '        free_dvector(crossp,0,s-1);\n',
                       '        free_dvector(qyp,0,s-1);\n',
                       '        free_dvector(qyp1,0,s-1);\n',
                       '        free_dvector(qyp2,0,s-1);\n',
                       '        free_dvector(qyp3,0,s-1);\n',
                       '        free_dvector(qyp4,0,s-1);\n',
                       '        free_dvector(qyp5,0,s-1);\n',
                       '        free_dvector(qyp6,0,s-1);\n',
                       '        free_dvector(qyp7,0,s-1);\n',
                       '        free_dvector(crosspt,0,s-1);\n',
                       '        free_dvector(qypt,0,s-1);\n',
                       '        free_dvector(qyp1t,0,s-1);\n',
                       '        free_dvector(qyp2t,0,s-1);\n',
                       '        free_dvector(qyp3t,0,s-1);\n',
                       '        free_dvector(qyp4t,0,s-1);\n',
                       '        free_dvector(qyp5t,0,s-1);\n',
                       '        free_dvector(qyp6t,0,s-1);\n',
                       '        free_dvector(qyp7t,0,s-1);\n',
                       #        printf("%s %s %s\\n", "The", species[j].name, "Cross section and quantum yield data are imported.");\n',
                       #        fprintf(fcheck, "%s %s %s\\n", "The", species[j].name, "Cross section and quantum yield data are imported.");\n',
                       #        for (j=0; j<NLAMBDA;j++) {fprintf(fcheck, "%lf %le %le %lf %lf\\n", wavelength[j], cross[i][j], crosst[i][j], qy[i][j], qyt[i][j]);}\n',
                       '    }\n',

                       # cross section of aerosols
                       '    double *crossp1, *crossp2, *crossp3;\n',
                       '    double crossw1[NLAMBDA], crossw2[NLAMBDA], crossw3[NLAMBDA];\n',
                       '    fp=fopen(AERRADFILE1,"r");\n',
                       '    fp1=fopen(AERRADFILE1,"r");\n',
                       '    s=LineNumber(fp, 1000);\n',
                       '    wavep=dvector(0,s-1);\n',
                       '    crossp1=dvector(0,s-1);\n',
                       '    crossp2=dvector(0,s-1);\n',
                       '    crossp3=dvector(0,s-1);\n',
                       '    k=0;\n',
                       '    while (fgets(dataline, 1000, fp1) != NULL ) {\n',
                       '        sscanf(dataline, "%lf %lf %lf %lf", wavep+k, crossp1+k, crossp2+k, crossp3+k);\n',
                       '        k=k+1; \n',
                       '    }\n',
                       '    fclose(fp);\n',
                       '    fclose(fp1);\n',
                       '    Interpolation(wavelength, NLAMBDA, crossw1, wavep, crossp1, s, 0);\n',
                       '    Interpolation(wavelength, NLAMBDA, crossw2, wavep, crossp2, s, 0);\n',
                       '    Interpolation(wavelength, NLAMBDA, crossw3, wavep, crossp3, s, 0);\n',
                       '    free_dvector(wavep,0,s-1);\n',
                       '    free_dvector(crossp1,0,s-1);\n',
                       '    free_dvector(crossp2,0,s-1);\n',
                       '    free_dvector(crossp3,0,s-1);\n',
                       '    for (i=' + str(iniz) + '; i<' + str(fine) + '; i++) {\n',
                       '        crossa[1][i] = crossw1[i];\n',
                       '        sinab[1][i]  = crossw2[i]/(crossw1[i]+1.0e-24);\n',
                       '        asym[1][i]   = crossw3[i];\n',
                       '    }\n',
                       '    fp=fopen(AERRADFILE2,"r");\n',
                       '    fp1=fopen(AERRADFILE2,"r");\n',
                       '    s=LineNumber(fp, 1000);\n',
                       '    wavep=dvector(0,s-1);\n',
                       '    crossp1=dvector(0,s-1);\n',
                       '    crossp2=dvector(0,s-1);\n',
                       '    crossp3=dvector(0,s-1);\n',
                       '    k=0;\n',
                       '    while (fgets(dataline, 1000, fp1) != NULL ) {\n',
                       '        sscanf(dataline, "%lf %lf %lf %lf", wavep+k, crossp1+k, crossp2+k, crossp3+k);\n',
                       '        k=k+1; \n',
                       '    }\n',
                       '    fclose(fp);\n',
                       '    fclose(fp1);\n',
                       '    Interpolation(wavelength, NLAMBDA, crossw1, wavep, crossp1, s, 0);\n',
                       '    Interpolation(wavelength, NLAMBDA, crossw2, wavep, crossp2, s, 0);\n',
                       '    Interpolation(wavelength, NLAMBDA, crossw3, wavep, crossp3, s, 0);\n',
                       '    free_dvector(wavep,0,s-1);\n',
                       '    free_dvector(crossp1,0,s-1);\n',
                       '    free_dvector(crossp2,0,s-1);\n',
                       '    free_dvector(crossp3,0,s-1);\n',
                       '    for (i=' + str(iniz) + '; i<' + str(fine) + '; i++) {\n',
                       '        crossa[2][i] = crossw1[i];\n',
                       '        sinab[2][i]  = crossw2[i]/(crossw1[i]+1.0e-24);\n',
                       '        asym[2][i]   = crossw3[i];\n',
                       '    }\n',
                       #    printf("%s\\n", "Cross sections of the aerosol are imported.");\n',
                       #    fprintf(fcheck, "%s\\n", "Cross sections of the aerosol are imported.");\n',
                       #    for (j=0; j<NLAMBDA;j++) {fprintf(fcheck, "%lf %e %e %f %f %f %f\\n", wavelength[j], crossa[1][j], crossa[2][j], sinab[1][j], sinab[2][j], asym[1][j], asym[2][j]);}\n',
                       #    fclose(fcheck);\n',

                       '    FILE *fim;\n',
                       '    double lll[324], ccc[324];\n',
                       '    lll[0] = 400.0;\n',
                       '    for (i=1; i < 324; i++) {\n',
                       '        lll[i]=lll[i-1] * (1.0+1.0 / 200.0);\n',
                       '    }\n',

                       '    char outaer1[1024];\n',
                       '    strcpy(outaer1, OUT_DIR);\n',
                       '    strcat(outaer1, "cross_H2O.dat");\n',
                       '    fim=fopen(outaer1,"r");\n',
                       '    for (j=1; j<=zbin; j++) {\n',
                       '        for (i=0; i < 324; i++) {fscanf(fim, "%le", ccc+i);}\n',
                       '        for (i=' + str(iniz) + '; i<' + str(fine) + '; i++) {\n',
                       '            Interpolation( & wavelength[i], 1, & cH2O[j][i], lll, ccc, 324, 2);\n',
                       '        }\n',
                       '    }\n',
                       '    fclose(fim);\n',

                       '    char outaer3[1024];\n',
                       '    strcpy(outaer3, OUT_DIR);\n',
                       '    strcat(outaer3, "geo_H2O.dat");\n',
                       '    fim = fopen(outaer3, "r");\n',
                       '    for (j=1; j<=zbin; j++) {\n',
                       '        for (i=0; i < 324; i++) {fscanf(fim, "%lf", ccc+i);}\n',
                       '        for (i=' + str(iniz) + '; i<' + str(fine) + '; i++) {\n',
                       '            Interpolation( & wavelength[i], 1, & gH2O[j][i], lll, ccc, 324, 2);\n',
                       '        }\n',
                       '    }\n',
                       '    fclose(fim);\n',

                       '    char outaer5[1024];\n',
                       '    strcpy(outaer5, OUT_DIR);\n',
                       '    strcat(outaer5, "albedo_H2O.dat");\n',
                       '    fim = fopen(outaer5, "r");\n',
                       '    for (j=1; j<=zbin; j++) {\n',
                       '        for (i=0; i < 324; i++) {fscanf(fim, "%lf", ccc+i);}\n',
                       '        for (i=' + str(iniz) + '; i<' + str(fine) + '; i++) {\n',
                       '            Interpolation( & wavelength[i], 1, & aH2O[j][i], lll, ccc, 324, 2);\n',
                       '        }\n',
                       '    }\n',
                       '    fclose(fim);\n',

                       # Geometric Albedo 9-point Gauss Quadruture
                       '    double cmiu[9]={-0.9681602395076261,-0.8360311073266358,-0.6133714327005904,-0.3242534234038089,0.0,0.3242534234038089,0.6133714327005904,0.8360311073266358,0.9681602395076261};\n',
                       '    double wmiu[9]={0.0812743883615744,0.1806481606948574,0.2606106964029354,0.3123470770400029,0.3302393550012598,0.3123470770400029,0.2606106964029354,0.1806481606948574,0.0812743883615744};\n',

                       '    double phase;\n',
                       '    phase = ' + str(self.param['phi']) + ';\n',  # Phase Angle, 0 zero geometric albedo
                       '    double lonfactor1, lonfactor2;\n',
                       '    double latfactor1, latfactor2;\n',
                       '    lonfactor1 = (PI-phase)*0.5;\n',
                       '    lonfactor2 = phase*0.5;\n',
                       '    latfactor1 = PI*0.5;\n',
                       '    latfactor2 = 0;\n',

                       '    double lat[9], lon[9];\n',
                       '    for (i=0; i<9; i++) {\n',
                       '        lat[i] = latfactor1*cmiu[i]+latfactor2;\n',
                       '        lon[i] = lonfactor1*cmiu[i]+lonfactor2;\n',
                       '    }\n',
                       '    double T0[zbin + 1];\n',
                       '    for (j=0; j <= zbin; j++) {\n',
                       '        T0[j]=0.0;\n',
                       '    }\n',

                       '    char uvrfile[1024];\n',

                       # Variation
                       '    double methaneexp[7]={0,0,0,0,0,0,0};\n',
                       '    int methaneid;\n',

                       '    double gmiu0, gmiu;\n',
                       '    double rout[NLAMBDA], gal[NLAMBDA];\n',
                       '    for (k=' + str(iniz) + '; k < ' + str(fine) + '; k++) {\n',
                       '        gal[k]=0;\n',
                       '    }\n',

                       '    for (j=1; j <= zbin; j++) {\n',
                       '        for (i=1; i <= NSP; i++) {\n',
                       '            xx1[j][i] = xx[j][i];\n',
                       '        }\n',
                       '    }\n',

                       '    strcpy(uvrfile, OUT_DIR);\n',
                       '    strcat(uvrfile, "Reflection_Phase.dat");\n',
                       '    for (i=0; i < 9; i++) {\n',
                       '        for (j=0; j < 9; j++) {\n',
                       '            gmiu0 = cos(lat[i]) * cos(lon[j]-phase);\n',
                       '            gmiu  = cos(lat[i]) * cos(lon[j]);\n',
                       '            if (fabs(gmiu0-gmiu) < 0.0000001) {\n',
                       '                gmiu=gmiu0+0.0000001;\n',
                       '            }\n',
                       # printf("%f %f %f %f\n", lat[i], lon[j], gmiu0, gmiu);
                       '            Reflection(xx1, T, stdcross, qysum, cross, crosst, uvrfile, gmiu0, gmiu, phase, rout, ' + str(iniz) + ', ' + str(fine) + ');\n',
                       '            for (k=' + str(iniz) + '; k < ' + str(fine) + '; k++) {\n',
                       '                gal[k] += wmiu[i] * wmiu[j] * rout[k] * gmiu0 * gmiu * cos(lat[i]) * latfactor1 * lonfactor1 / PI;\n',
                       '            }\n',
                       '        }\n',
                       '    }\n',

                       # print out spectra
                       '    char outag[1024];\n',
                       '    strcpy(outag, OUT_DIR);\n',
                       '    strcat(outag, "PhaseA.dat");\n',
                       '    fp = fopen(outag, "w");\n',
                       '    for (i=' + str(iniz) + '; i < ' + str(fine) + '; i++) {\n',
                       '        fprintf(fp, "%f\t", wavelength[i]);\n',
                       '        fprintf(fp, "%e\t", gal[i]);\n',
                       '        fprintf(fp, "\\n");\n',
                       '    }\n',
                       '    fclose(fp);\n',

                       # Clean up
                       '    free_dmatrix(cross, 1, nump, 0, NLAMBDA - 1);\n',
                       '    free_dmatrix(qy, 1, nump, 0, NLAMBDA - 1);\n',
                       '    free_dmatrix(opacH2O, 1, zbin, 0, NLAMBDA - 1);\n',
                       '    free_dmatrix(opacNH3, 1, zbin, 0, NLAMBDA - 1);\n',
                       '    free_dmatrix(opacCH4, 1, zbin, 0, NLAMBDA - 1);\n',
                       '    free_dmatrix(opacH2S, 1, zbin, 0, NLAMBDA - 1);\n',
                       '    free_dmatrix(opacSO2, 1, zbin, 0, NLAMBDA - 1);\n',
                       '    free_dmatrix(opacCO2, 1, zbin, 0, NLAMBDA - 1);\n',
                       '    free_dmatrix(opacCO, 1, zbin, 0, NLAMBDA - 1);\n',
                       '    free_dmatrix(opacO2, 1, zbin, 0, NLAMBDA - 1);\n',
                       '    free_dmatrix(opacO3, 1, zbin, 0, NLAMBDA - 1);\n',
                       '    free_dmatrix(opacN2O, 1, zbin, 0, NLAMBDA - 1);\n',
                       '    free_dmatrix(opacN2, 1, zbin, 0, NLAMBDA - 1);\n',
                       #    free_dmatrix(opacOH, 1, zbin, 0, NLAMBDA - 1);\n',
                       #    free_dmatrix(opacH2CO, 1, zbin, 0, NLAMBDA - 1);\n',
                       #    free_dmatrix(opacH2O2, 1, zbin, 0, NLAMBDA - 1);\n',
                       #    free_dmatrix(opacHO2, 1, zbin, 0, NLAMBDA - 1);\n',
                       #    free_dmatrix(opacC2H2, 1, zbin, 0, NLAMBDA - 1);\n',
                       #    free_dmatrix(opacC2H4, 1, zbin, 0, NLAMBDA - 1);\n',
                       #    free_dmatrix(opacC2H6, 1, zbin, 0, NLAMBDA - 1);\n',
                       #    free_dmatrix(opacHCN, 1, zbin, 0, NLAMBDA - 1);\n',
                       #    free_dmatrix(opacCH2O2, 1, zbin, 0, NLAMBDA - 1);\n',
                       #    free_dmatrix(opacHNO3, 1, zbin, 0, NLAMBDA - 1);\n',
                       #    free_dmatrix(opacNO, 1, zbin, 0, NLAMBDA - 1);\n',
                       #    free_dmatrix(opacNO2, 1, zbin, 0, NLAMBDA - 1);\n',
                       #    free_dmatrix(opacOCS, 1, zbin, 0, NLAMBDA - 1);\n',
                       #    free_dmatrix(opacHF, 1, zbin, 0, NLAMBDA - 1);\n',
                       #    free_dmatrix(opacHCl, 1, zbin, 0, NLAMBDA - 1);\n',
                       #    free_dmatrix(opacHBr, 1, zbin, 0, NLAMBDA - 1);\n',
                       #    free_dmatrix(opacHI, 1, zbin, 0, NLAMBDA - 1);\n',
                       #    free_dmatrix(opacClO, 1, zbin, 0, NLAMBDA - 1);\n',
                       #    free_dmatrix(opacHClO, 1, zbin, 0, NLAMBDA - 1);\n',
                       #    free_dmatrix(opacHBrO, 1, zbin, 0, NLAMBDA - 1);\n',
                       #    free_dmatrix(opacPH3, 1, zbin, 0, NLAMBDA - 1);\n',
                       #    free_dmatrix(opacCH3Cl, 1, zbin, 0, NLAMBDA - 1);\n',
                       #    free_dmatrix(opacCH3Br, 1, zbin, 0, NLAMBDA - 1);\n',
                       #    free_dmatrix(opacDMS, 1, zbin, 0, NLAMBDA - 1);\n',
                       #    free_dmatrix(opacCS2, 1, zbin, 0, NLAMBDA - 1);\n',
                       '    free_dmatrix(xx, 1, zbin, 1, NSP);\n',
                       '    free_dmatrix(xx1, 1, zbin, 1, NSP);\n',
                       '    free_dmatrix(xx2, 1, zbin, 1, NSP);\n',
                       '    free_dmatrix(xx3, 1, zbin, 1, NSP);\n',
                       '    free_dmatrix(xx4, 1, zbin, 1, NSP);\n',

                       '}\n']

        with open(self.c_code_directory + 'core_' + str(self.process) + '.c', 'w') as file:
            for riga in c_core_file:
                file.write(riga)

    def __run_c_code(self):
        self.__par_c_file()
        self.__core_c_file()
        os.chdir(self.c_code_directory)
        if platform.system() == 'Darwin':
            os.system('clang -Wno-nullability-completeness -o ' + str(self.process) + ' core_' + str(self.process) + '.c -lm')
        else:
            os.system('gcc -o ' + str(self.process) + ' core_' + str(self.process) + '.c -lm')
        while not os.path.exists(self.c_code_directory + str(self.process)):
            pass
        time.sleep(2)
        os.system('chmod +rwx ' + str(self.process))
        os.system('./' + str(self.process))
        os.chdir(self.working_dir)
        time1 = time.time()
        broken = False
        while not os.path.exists(self.c_code_directory + 'Result/Retrieval_' + str(self.process) + '/PhaseA.dat'):
            time2 = time.time()
            if time2 - time1 > 600:
                broken = True
                break
            else:
                pass
        os.system('rm -rf ' + self.c_code_directory + str(self.process))
        os.system('rm -rf ' + self.c_code_directory + 'core_' + str(self.process) + '.c')
        os.system('rm -rf ' + self.c_code_directory + 'par_' + str(self.process) + '.h')
        if not broken:
            albedo = np.loadtxt(self.c_code_directory + 'Result/Retrieval_' + str(self.process) + '/PhaseA.dat')
        else:
            albedo = np.zeros((1000, 2))
            albedo[:, 1] = np.nan
            albedo[:, 0] = np.linspace(self.param['min_wl'], self.param['max_wl'], num=1000)
        if self.retrieval:
            os.system('rm -rf ' + self.c_code_directory + 'Result/Retrieval_' + str(self.process) + '/')
        else:
            if self.canc_metadata:
                os.system('rm -rf ' + self.c_code_directory + 'Result/Retrieval_' + str(self.process) + '/')
            else:
                pass

        return albedo[:, 0], albedo[:, 1]

    def run_forward(self):
        self.__run_structure()
        alb_wl, alb = self.__run_c_code()

        alb_wl *= 10. ** (-3.)

        return alb_wl, alb


class FORWARD_DATASET:
    """
    Interpolate a precomputed dataset (built with GEN_DATASET) at desired parameters.

    Usage
    -----
    mod = FORWARD_DATASET(param, dataset_dir)
    alb_wl, alb = mod.run_forward()

    Returns
    -------
    alb_wl : 1-D array
        Wavelength grid loaded from the dataset wave_file (micron).
    alb : 1-D array
        Interpolated spectrum (same quantity stored in the dataset; typically albedo).
    """

    # Parameters that are sampled/stored in log10-space in the dataset
    _LOG10_KEYS = {"Pw_top", "cldw_depth", "CR_H2O", "Pa_top", "clda_depth", "CR_NH3", "p_size"}

    def __init__(self, par, dataset_dir):
        self.param = copy.deepcopy(par)
        # Ensure core paths and spectrum defaults are present
        self.dataset_dir = dataset_dir

    def _load_design_matrix(self):
        csv_path = os.path.join(self.dataset_dir, 'dataset.csv')
        meta_path = os.path.join(self.dataset_dir, 'dataset_meta.json')

        if not os.path.isfile(csv_path):
            raise FileNotFoundError('dataset.csv not found in: ' + self.dataset_dir)
        if not os.path.isfile(meta_path):
            raise FileNotFoundError('dataset_meta.json not found in: ' + self.dataset_dir)

        # Read header to get column order (index, ...parameters...)
        with open(csv_path, 'r') as f:
            header = f.readline().strip()
        cols = [h.strip() for h in header.split(',')]
        if len(cols) < 2 or cols[0] != 'index':
            raise ValueError('Invalid dataset.csv header; first column must be "index"')

        # Load numeric data (may be 1-D if only one row)
        data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        index = data[:, 0].astype(int)
        X = data[:, 1:]

        # Cross-check with meta (best-effort)
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            meta_cols = meta.get('columns')
            if meta_cols is not None and list(meta_cols) != cols:
                raise ValueError('dataset_meta.json columns do not match dataset.csv header')
        except Exception:
            # Do not fail hard if meta is minimally informative
            pass

        return cols[1:], index, X

    def _target_from_param(self, colnames):
        gps = self.param.get('gas_par_space')
        if gps not in ('volume_mixing_ratio', 'vmr', 'partial_pressure'):
            raise ValueError('Unsupported gas_par_space for interpolation: ' + str(gps))

        xtgt = []
        for cname in colnames:
            if cname.endswith('_range'):
                key = cname[:-6]
                if key not in self.param:
                    raise KeyError('Parameter "' + key + '" required by dataset missing in self.param')
                val = self.param[key]
                if key in self._LOG10_KEYS:
                    if val is None:
                        raise ValueError('Parameter "' + key + '" is None; cannot take log10')
                    if val <= 0:
                        raise ValueError('Parameter "' + key + '" must be > 0 for log10 mapping')
                    val = np.log10(val)
                xtgt.append(float(val))
            else:
                # Molecule dimension
                mol = cname
                if gps in ('volume_mixing_ratio', 'vmr'):
                    v = self.param.get('vmr_' + mol)
                    if v is None:
                        raise KeyError('Missing self.param["vmr_' + mol + '"] for vmr gas_par_space')
                    xtgt.append(float(np.log10(v[-1])))
                elif gps == 'partial_pressure':
                    v = self.param.get('vmr_' + mol)
                    P0 = self.param.get('P0')
                    if v is None or P0 is None:
                        raise KeyError('Missing vmr_' + mol + ' or P0 for partial_pressure gas_par_space')
                    if v * P0 <= 0:
                        raise ValueError('vmr_' + mol + ' * P0 must be > 0 for log10 mapping')
                    xtgt.append(float(np.log10(v * P0)[-1]))
        return np.array(xtgt, dtype=float)

    def _load_spectra_matrix(self, indices):
        # Load spectra for all sample indices listed in dataset.csv
        Y = None
        wave_file_id = None
        for k, idx in enumerate(indices):
            fname = os.path.join(self.dataset_dir, f'sample_{int(idx):07d}.json')
            if not os.path.isfile(fname):
                raise FileNotFoundError('Sample file missing: ' + fname)
            with open(fname, 'r') as f:
                rec = json.load(f)
            spv = np.asarray(rec['spectrum'], dtype=float)
            if Y is None:
                Y = np.empty((len(indices), spv.size), dtype=float)
            Y[k, :] = spv

            if wave_file_id is None:
                wave_file_id = rec.get('wavelength')
        if Y is None:
            raise RuntimeError('No spectra loaded from dataset directory')

        return Y, wave_file_id

    def _load_wavelength_grid(self, wave_file_id):
        # Resolve the wavelength bins file from package dir
        if wave_file_id is None:
            raise ValueError('Wave file identifier missing from dataset samples')
        bins_path = os.path.join(self.param['pkg_dir'], 'forward_mod', 'Data', 'wl_bins', wave_file_id + '.dat')
        try:
            spectrum = np.loadtxt(bins_path)
        except Exception as e:
            raise FileNotFoundError('Cannot load wavelength grid file: ' + bins_path + ' (' + str(e) + ')')

        # Accept 1D (wl), 2D (low, high), or 3+ columns with wl at column 2
        if spectrum.ndim == 1:
            wl = spectrum + 0.0
        elif spectrum.shape[1] == 2:
            wl = np.mean(np.array([spectrum[:, 0], spectrum[:, 1]]).T, axis=1)
        else:
            wl = spectrum[:, 2] + 0.0
        return wl

    def run_forward(self):
        # 1) Load design matrix
        colnames, idx, X = self._load_design_matrix()

        # 2) Build target vector from current parameters
        xtgt = self._target_from_param(colnames)

        # 3) Validate target within dataset bounds
        mins = np.nanmin(X, axis=0)
        maxs = np.nanmax(X, axis=0)
        for i, (mn, mx) in enumerate(zip(mins, maxs)):
            if not (mn <= xtgt[i] <= mx):
                raise ValueError('Target parameter "' + str(colnames[i]) + '"=' + str(xtgt[i]) +
                                 ' out of dataset range [' + str(mn) + ', ' + str(mx) + ']')

        # 4) Load spectra matrix and wavelength id
        # For large datasets, load only k-nearest samples in parameter space to the target
        n_samples = X.shape[0]
        dim = X.shape[1]
        file_to_open = 2 ** dim

        if n_samples > file_to_open:
            # Compute squared distances to target and select k nearest
            d2 = np.sum((X - xtgt) ** 2, axis=1)
            k = min(file_to_open, n_samples)
            # Ensure we have at least dim+1 points for simplex interpolation when possible
            k = max(min(n_samples, k), min(dim + 1, n_samples))
            nn_idx = np.argpartition(d2, k - 1)[:k]
            X_use = X[nn_idx]
            idx_use = idx[nn_idx]
        else:
            X_use = X
            idx_use = idx

        Y, wave_file_id = self._load_spectra_matrix(idx_use)

        # 5) Check if inside convex hull and interpolate
        # Use Delaunay once and barycentric linear interpolation for speed
        # Build Delaunay on the working set and interpolate
        # If target lies outside this subset hull or triangulation fails, fall back to nearest
        y = None
        try:
            tri = sp.spatial.Delaunay(X_use)
            simplex = tri.find_simplex(xtgt)
            if simplex >= 0:
                interp = sp.interpolate.LinearNDInterpolator(tri, Y)
                y = interp(xtgt)
        except Exception:
            y = None

        if y is None or np.any(~np.isfinite(y)):
            # Fallback to nearest interpolation on the working set
            y = sp.interpolate.NearestNDInterpolator(X_use, Y)(xtgt)

        alb_wl = self._load_wavelength_grid(wave_file_id)
        alb = np.asarray(y, dtype=float)

        return alb_wl, alb[0]


def forward(parameters_dictionary, evaluation=None, phi=None, n_obs=None, retrieval_mode=True, core_number=None, albedo_calc=False, fp_over_fs=False, canc_metadata=False):
    param = copy.deepcopy(parameters_dictionary)

    if evaluation is not None:
        if param['fit_p0'] or param['gas_par_space'] == 'partial_pressure':
            param['P0'] = evaluation['P0']
        if param['fit_wtr_cld']:
            param['Pw_top'] = evaluation['pH2O']
            param['cldw_depth'] = evaluation['dH2O']
            param['CR_H2O'] = evaluation['crH2O']
        if param['fit_amm_cld']:
            param['Pa_top'] = evaluation['pNH3']
            param['clda_depth'] = evaluation['dNH3']
            param['CR_NH3'] = evaluation['crNH3']

        for mol in param['fit_molecules']:
            param['vmr_' + mol] = evaluation[mol]
        if param['gas_fill'] is not None:
            param['vmr_' + param['gas_fill']] = evaluation[param['gas_fill']]

        if param['fit_ag']:
            if param['surface_albedo_parameters'] == int(1):
                param['Ag'] = evaluation['ag']
            elif param['surface_albedo_parameters'] == int(3):
                for surf_alb in [1, 2]:
                    param['Ag' + str(surf_alb)] = evaluation['ag' + str(surf_alb)]
                param['Ag_x1'] = evaluation['ag_x1']
            elif param['surface_albedo_parameters'] == int(5):
                for surf_alb in [1, 2, 3]:
                    param['Ag' + str(surf_alb)] = evaluation['ag' + str(surf_alb)]
                param['Ag_x1'] = evaluation['ag_x1']
                param['Ag_x2'] = evaluation['ag_x2']

        if param['fit_T']:
            param['Tp'] = evaluation['Tp']
        if param['fit_g'] and param['fit_Mp'] and not param['fit_Rp']:
            param['gp'] = (10. ** (evaluation['gp'] - 2.0))                                                                     # g is in m/s2 but it was defined in cgs
            param['Mp'] = evaluation['Mp']                                                                                      # Mp is in M_jup
            param['Rp'] = (np.sqrt((const.G.value * const.M_jup.value * param['Mp']) / param['gp'])) / const.R_jup.value        # Rp is in R_jup
        elif param['fit_g'] and param['fit_Rp'] and not param['fit_Mp']:
            param['gp'] = (10. ** (evaluation['gp'] - 2.0))                                                                     # g is in m/s2 but it was defined in cgs
            param['Rp'] = evaluation['Rp']                                                                                      # Rp is in R_jup
            param['Mp'] = ((param['gp'] * ((param['Rp'] * const.R_jup.value) ** 2.)) / const.G.value) / const.M_jup.value       # Mp is in M_jup
        elif param['fit_Mp'] and param['fit_Rp'] and not param['fit_g']:
            param['Mp'] = evaluation['Mp']                                                                                      # Mp is in M_jup
            param['Rp'] = evaluation['Rp']                                                                                      # Rp is in R_jup
            param['gp'] = (const.G.value * const.M_jup.value * param['Mp']) / ((const.R_jup.value * param['Rp']) ** 2.)         # g is in m/s2
        elif param['fit_g'] and not param['fit_Mp'] and not param['fit_Rp'] and param['Mp'] is not None:
            param['gp'] = (10. ** (evaluation['gp'] - 2.0))                                                                     # g is in m/s2 but it was defined in cgs
            param['Rp'] = (np.sqrt((const.G.value * const.M_jup.value * param['Mp']) / param['gp'])) / const.R_jup.value        # Rp is in R_jup
        elif param['fit_Rp'] and not param['fit_Mp'] and not param['fit_g'] and param['Mp'] is not None:
            param['Rp'] = evaluation['Rp']                                                                                      # Rp is in R_jup
            if param['Mp_err'] is not None and param['Mp_prior_type'] == 'random_error':
                param['Mp'] = np.random.normal(param['Mp_orig'], param['Mp_err'])
            else:
                param['Mp'] = param['Mp_orig'] + 0.0
            param['gp'] = (const.G.value * const.M_jup.value * param['Mp']) / ((const.R_jup.value * param['Rp']) ** 2.)         # g is in m/s2
        elif not param['fit_g'] and not param['fit_Mp'] and not param['fit_Rp']:
            if not param['Mp_provided']:
                param['Mp'] = ((param['gp'] * ((param['Rp'] * const.R_jup.value) ** 2.)) / const.G.value) / const.M_jup.value   # Mp is in M_jup
            if not param['Rp_provided']:
                param['Rp'] = (np.sqrt((const.G.value * const.M_jup.value * param['Mp']) / param['gp'])) / const.R_jup.value    # Rp is in R_jup
            if not param['gp_provided']:
                param['gp'] = (const.G.value * const.M_jup.value * param['Mp']) / ((const.R_jup.value * param['Rp']) ** 2.)     # g is in m/s2

        if param['fit_p_size']:
            param['p_size'] = evaluation['p_size']

        if param['fit_phi']:
            param['phi'] = evaluation['phi']

    if phi is not None:
        param['phi'] = phi

    param['core_number'] = core_number

    if param['gas_par_space'] == 'partial_pressure' and np.log10(param['P0']) < 0.0:
        param['P0'] = 1.1
    param['P'] = 10. ** np.arange(0.0, np.log10(param['P0']) + 0.01, step=0.01)
    if param['fit_amm_cld']:
        param['vmr_NH3'] = cloud_pos(param, condensed_gas='NH3')
        param = adjust_VMR(param, all_gases=param['adjust_VMR_gases'], condensed_gas='NH3')
    param['vmr_H2O'] = cloud_pos(param, condensed_gas='H2O')
    param = adjust_VMR(param, all_gases=param['adjust_VMR_gases'], condensed_gas='H2O')
    if param['O3_earth']:
        param['vmr_O3'] = ozone_earth_mask(param)
    param = calc_mean_mol_mass(param)

    if not retrieval_mode and param['verbose']:
        if n_obs == 0 or n_obs is None:
            if albedo_calc and not fp_over_fs:
                print('Calculating the planetary albedo as function of wavelength')
            elif fp_over_fs and not albedo_calc:
                print('Calculating the contrast ratio as function of wavelength')
            else:
                print('Calculating the planetary flux as function of wavelength')

    if param['physics_model'] == 'radiative_transfer':
        mod = FORWARD_MODEL(param, retrieval=retrieval_mode, canc_metadata=canc_metadata)
        alb_wl, alb = mod.run_forward()
        wl, model = model_finalizzation(param, alb_wl, alb, planet_albedo=albedo_calc, fp_over_fs=fp_over_fs, n_obs=n_obs)
    elif param['physics_model'] == 'dataset':
        mod = FORWARD_DATASET(param, dataset_dir=param['dataset_dir'])
        wl, fpfs = mod.run_forward()
        wl, model = model_finalizzation(param, alb_wl, fpfs, phys_mod=param['physics_model'], n_obs=n_obs)
    elif param['physics_model'] == 'AI_model':
        pass # to be implemented
    else:
        raise ValueError('Unknown physics_model: ' + str(param['physics_model']))

    if retrieval_mode:
        return model
    else:
        return wl, model
