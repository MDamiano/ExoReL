from .__basics import *
from .__utils import *
from pathlib import Path


class RADIATIVE_TRANSFER_C:
    def __init__(self, param, retrieval=True, canc_metadata=False):
        self.param = copy.deepcopy(param)
        self.process = str(self.param['core_number']) + str(random.randint(0, 100000)) + alphabet() + alphabet() + alphabet() + str(random.randint(0, 100000))
        self.package_dir = param['pkg_dir']
        self.retrieval = retrieval
        self.canc_metadata = canc_metadata
        self.hazes_calc = param['hazes']
        self.c_code_directory = self.package_dir + 'forward_mod/'
        try:
            self.working_dir = param['wkg_dir']
        except KeyError:
            self.working_dir = os.getcwd()

    def __surface_structure(self):
        wl_grid = np.asarray(self.param['wl_C_grid'])
        self.surf_alb = np.zeros(len(wl_grid))
        if self.param['surface_albedo_parameters'] == int(1):
            self.surf_alb += self.param['Ag']
        elif self.param['surface_albedo_parameters'] == int(3):
            left_mask = wl_grid < self.param['Ag_x1']
            self.surf_alb[left_mask] = self.param['Ag1']
            self.surf_alb[~left_mask] = self.param['Ag2']
        elif self.param['surface_albedo_parameters'] == int(5):
            left_mask = wl_grid < self.param['Ag_x1']
            middle_mask = (wl_grid >= self.param['Ag_x1']) & (wl_grid < self.param['Ag_x2'])
            right_mask = wl_grid >= self.param['Ag_x2']
            self.surf_alb[left_mask] = self.param['Ag1']
            self.surf_alb[middle_mask] = self.param['Ag2']
            self.surf_alb[right_mask] = self.param['Ag3']

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

        deltaP = 0.001 * 2.65495471  # assume super saturation to be 0.1% at 220 K

        g = self.param['gp'] + 0.0

        # Set up pressure grid
        P = self.param['P'] + 0.0  # in Pascal

        # Temperature profile
        T = self.param['T'] * np.ones(len(P)) # in K

        if self.param['fit_wtr_cld'] and 'H2O' in self.param['fit_molecules']:
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
                    r0, _, r2, VP = particlesizef(g, T[i], P[i], self.param['mean_mol_weight'][i], self.param['mm']['H2O'], self.param['KE'], deltaP)
                    if self.param['fit_p_size'] and self.param['p_size_type'] == 'factor':
                        particlesize[i] = r2 * self.param['p_size']
                    else:
                        particlesize[i] = r2 + 0.0

        if self.param['fit_amm_cld'] and 'NH3' in self.param['fit_molecules']:
            cloudden_nh3 = 1.0e-36 * np.ones(len(P))
            for i in range(len(P) - 2, -1, -1):
                cloudden_nh3[i] = max(abs(self.param['vmr_NH3'][i] - self.param['vmr_NH3'][i + 1]) * 0.017 * P[i] / const.R.value / T[i], 1e-25)  # kg/m^3, g/L
            particlesize_nh3 = 1.0e-36 * np.ones(len(P))

            if self.param['fit_p_size'] and self.param['p_size_type'] == 'constant':
                particlesize_nh3 = self.param['p_size'] * np.ones(len(P))
            else:
                for i in range(len(P) - 2, -1, -1):
                    deltaP_nh3 = P[i] * abs(self.param['vmr_NH3'][i] - self.param['vmr_NH3'][i + 1])
                    r0, _, r2, VP = particlesizef(g, T[i], P[i], self.param['mean_mol_weight'][i], self.param['mm']['NH3'], self.param['KE'], deltaP_nh3)
                    if self.param['fit_p_size'] and self.param['p_size_type'] == 'factor':
                        particlesize_nh3[i] = r2 * self.param['p_size']
                    else:
                        particlesize_nh3[i] = r2 + 0.0

        # Calculate the height
        P = P[::-1]
        T = T[::-1]
        if self.param['fit_wtr_cld'] and 'H2O' in self.param['fit_molecules']:
            cloudden = cloudden[::-1]
            particlesize = particlesize[::-1]
        if self.param['fit_amm_cld'] and 'NH3' in self.param['fit_molecules']:
            cloudden_nh3 = cloudden_nh3[::-1]
            particlesize_nh3 = particlesize_nh3[::-1]
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
            idx_cloud_sets = []
            if self.param['fit_wtr_cld'] and 'H2O' in f:
                idx_cloud_sets.append(np.where(np.diff(f['H2O']) != 0.0)[0] + 1)
            if self.param['fit_amm_cld'] and 'NH3' in f:
                idx_cloud_sets.append(np.where(np.diff(f['NH3']) != 0.0)[0] + 1)
            if idx_cloud_sets:
                idx_cloud_layers = np.unique(np.concatenate(idx_cloud_sets))
            else:
                idx_cloud_layers = np.array([], dtype=int)
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
            if self.param['fit_wtr_cld']:
                np.savetxt(self.outdir + 'watermix.dat', f['H2O'])
                np.savetxt(self.outdir + 'particlesize.dat', particlesize)
                np.savetxt(self.outdir + 'cloudden.dat', cloudden)
            if self.param['fit_amm_cld']:
                np.savetxt(self.outdir + 'ammoniamix.dat', f['NH3'])
                np.savetxt(self.outdir + 'particlesize_nh3.dat', particlesize_nh3)
                np.savetxt(self.outdir + 'cloudden_nh3.dat', cloudden_nh3)

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

        if self.param['fit_wtr_cld']:
            tck = interp1d(Z, np.log(cloudden))
            cloudden = np.exp(tck(zl))

            tck = interp1d(Z, np.log(particlesize))
            particlesize = np.exp(tck(zl))

        if self.param['fit_amm_cld']:
            tck = interp1d(Z, np.log(cloudden_nh3))
            cloudden_nh3 = np.exp(tck(zl))

            tck = interp1d(Z, np.log(particlesize_nh3))
            particlesize_nh3 = np.exp(tck(zl))

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
        if self.param['fit_wtr_cld']:
            cro_h2o = np.zeros((len(zl), 324))
            alb_h2o = np.ones((len(zl), 324))
            geo_h2o = np.zeros((len(zl), 324))
        
        if self.param['fit_amm_cld']:
            cro_nh3 = np.zeros((len(zl), 324))
            alb_nh3 = np.ones((len(zl), 324))
            geo_nh3 = np.zeros((len(zl), 324))

        #    opacity
        sig = 2
        for j in range(0, len(zl)):
            if self.param['fit_wtr_cld']:
                r2 = particlesize[j]
                if cloudden[j] < 1e-16:
                    pass
                else:
                    r0 = r2 * np.exp(-np.log(sig) ** 2.)
                    if self.param['wtr_cld_type'] == 'mixed' and self.param['PT_profile_type'] == 'parametric':
                        if tl[j] < 273.15: # ice
                            VP = 4. * math.pi / 3. * ((r2 * 1.0e-6 * np.exp(0.5 * np.log(sig) ** 2.)) ** 3.) * 1.0e+6 * 0.92  # g
                            for indi in range(0, 324):
                                tck = interp1d(np.log10(self.param['H2OL_r']), np.log10(self.param['H2OL_c_ice'][:, indi]))
                                temporaneo = tck(np.log10(max(0.01, min(r0, 100))))
                                cro_h2o[j, indi] = cloudden[j] / VP * 1.0e-3 * (10. ** temporaneo)  # cm-1
                                tck = interp1d(np.log10(self.param['H2OL_r']), self.param['H2OL_a_ice'][:, indi])
                                alb_h2o[j, indi] = tck(np.log10(max(0.01, min(r0, 100))))
                                tck = interp1d(np.log10(self.param['H2OL_r']), self.param['H2OL_g_ice'][:, indi])
                                geo_h2o[j, indi] = tck(np.log10(max(0.01, min(r0, 100))))
                        else: # liquid
                            VP = 4. * math.pi / 3. * ((r2 * 1.0e-6 * np.exp(0.5 * np.log(sig) ** 2.)) ** 3.) * 1.0e+6 * 1.0  # g
                            for indi in range(0, 324):
                                tck = interp1d(np.log10(self.param['H2OL_r']), np.log10(self.param['H2OL_c_liquid'][:, indi]))
                                temporaneo = tck(np.log10(max(0.01, min(r0, 100))))
                                cro_h2o[j, indi] = cloudden[j] / VP * 1.0e-3 * (10. ** temporaneo)  # cm-1
                                tck = interp1d(np.log10(self.param['H2OL_r']), self.param['H2OL_a_liquid'][:, indi])
                                alb_h2o[j, indi] = tck(np.log10(max(0.01, min(r0, 100))))
                                tck = interp1d(np.log10(self.param['H2OL_r']), self.param['H2OL_g_liquid'][:, indi])
                                geo_h2o[j, indi] = tck(np.log10(max(0.01, min(r0, 100))))
                    else:  # liquid or ice
                        if self.param['wtr_cld_type'] == 'liquid':  # liquid
                            VP = 4. * math.pi / 3. * ((r2 * 1.0e-6 * np.exp(0.5 * np.log(sig) ** 2.)) ** 3.) * 1.0e+6 * 1.0  # g
                        else:  # ice
                            VP = 4. * math.pi / 3. * ((r2 * 1.0e-6 * np.exp(0.5 * np.log(sig) ** 2.)) ** 3.) * 1.0e+6 * 0.92  # g
                        for indi in range(0, 324):
                            tck = interp1d(np.log10(self.param['H2OL_r']), np.log10(self.param['H2OL_c'][:, indi]))
                            temporaneo = tck(np.log10(max(0.01, min(r0, 100))))
                            cro_h2o[j, indi] = cloudden[j] / VP * 1.0e-3 * (10. ** temporaneo)  # cm-1
                            tck = interp1d(np.log10(self.param['H2OL_r']), self.param['H2OL_a'][:, indi])
                            alb_h2o[j, indi] = tck(np.log10(max(0.01, min(r0, 100))))
                            tck = interp1d(np.log10(self.param['H2OL_r']), self.param['H2OL_g'][:, indi])
                            geo_h2o[j, indi] = tck(np.log10(max(0.01, min(r0, 100))))
            
            if self.param['fit_amm_cld']:
                r2 = particlesize_nh3[j]
                if cloudden_nh3[j] < 1e-16:
                    pass
                else:
                    r0 = r2 * np.exp(-np.log(sig) ** 2.)  # micron
                    VP = 4. * math.pi / 3. * ((r2 * 1.0E-6 * np.exp(0.5 * (np.log(sig) ** 2.))) ** 3.) * 1.0E+6 * 0.87  # g
                    for indi in range(0, 324):
                        tck = interp1d(np.log10(self.param['NH3_r']), np.log10(self.param['NH3_c'][:, indi]))
                        temporaneo = tck(np.log10(max(0.01, min(r0, 100))))
                        cro_nh3[j, indi] = cloudden_nh3[j] / VP * 1.0e-3 * (10. ** temporaneo)  # cm-1
                        tck = interp1d(np.log10(self.param['NH3_r']), self.param['NH3_a'][:, indi])
                        alb_nh3[j, indi] = tck(np.log10(max(0.01, min(r0, 100))))
                        tck = interp1d(np.log10(self.param['NH3_r']), self.param['NH3_g'][:, indi])
                        geo_nh3[j, indi] = tck(np.log10(max(0.01, min(r0, 100))))

        if self.param['fit_wtr_cld']:
            with open(self.outdir + 'cross_H2O.dat', 'w') as file:
                for j in range(0, len(zl)):
                    for indi in range(0, 324):
                        file.write("{:.6e}".format(cro_h2o[j, indi]) + '\t')
                    file.write('\n')

            with open(self.outdir + 'albedo_H2O.dat', 'w') as file:
                for j in range(0, len(zl)):
                    for indi in range(0, 324):
                        file.write("{:.6e}".format(alb_h2o[j, indi]) + '\t')
                    file.write('\n')

            with open(self.outdir + 'geo_H2O.dat', 'w') as file:
                for j in range(0, len(zl)):
                    for indi in range(0, 324):
                        file.write("{:.6e}".format(geo_h2o[j, indi]) + '\t')
                    file.write('\n')
        
        if self.param['fit_amm_cld']:
            with open(self.outdir + 'cross_NH3.dat', 'w') as file:
                for j in range(0, len(zl)):
                    for indi in range(0, 324):
                        file.write("{:.6e}".format(cro_nh3[j, indi]) + '\t')
                    file.write('\n')
            
            with open(self.outdir + 'albedo_NH3.dat', 'w') as file:
                for j in range(0, len(zl)):
                    for indi in range(0, 324):
                        file.write("{:.6e}".format(alb_nh3[j, indi]) + '\t')
                    file.write('\n')
            
            with open(self.outdir + 'geo_NH3.dat', 'w') as file:
                for j in range(0, len(zl)):
                    for indi in range(0, 324):
                        file.write("{:.6e}".format(geo_nh3[j, indi]) + '\t')
                    file.write('\n')

    def __run_structure(self):
        os.chdir(self.c_code_directory)
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
                      '#define PSURFEM              1.0\n',  # Planet Surface Emissivity
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
                      # '#define TPMODE               1\n',  # 1: import data from a ZTP list
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
                      '#define TINTSET              ' + str(self.param['Tint']) + '\n',  # Internal Heat Temperature
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
                       '#include <stdbool.h>\n',
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
                       '    }\n']

        if self.param['fit_wtr_cld']:
            c_core_file += ['    char outaer1[1024];\n',
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
                            '    fclose(fim);\n'
                            '    bool wtr_cld = true;\n']
        else:
            c_core_file += ['    bool wtr_cld = false;\n']
        
        if self.param['fit_amm_cld']:
            c_core_file += ['    char outaer2[1024];\n',
                            '    strcpy(outaer2, OUT_DIR);\n',
                            '    strcat(outaer2, "cross_NH3.dat");\n',
                            '    fim=fopen(outaer2,"r");\n',
                            '    for (j=1; j<=zbin; j++) {\n',
                            '        for (i=0; i < 324; i++) {fscanf(fim, "%le", ccc+i);}\n',
                            '        for (i=' + str(iniz) + '; i<' + str(fine) + '; i++) {\n',
                            '            Interpolation( & wavelength[i], 1, & cNH3[j][i], lll, ccc, 324, 2);\n',
                            '        }\n',
                            '    }\n',
                            '    fclose(fim);\n',

                            '    char outaer4[1024];\n',
                            '    strcpy(outaer4, OUT_DIR);\n',
                            '    strcat(outaer4, "geo_NH3.dat");\n',
                            '    fim = fopen(outaer4, "r");\n',
                            '    for (j=1; j<=zbin; j++) {\n',
                            '        for (i=0; i < 324; i++) {fscanf(fim, "%lf", ccc+i);}\n',
                            '        for (i=' + str(iniz) + '; i<' + str(fine) + '; i++) {\n',
                            '            Interpolation( & wavelength[i], 1, & gNH3[j][i], lll, ccc, 324, 2);\n',
                            '        }\n',
                            '    }\n',
                            '    fclose(fim);\n',

                            '    char outaer6[1024];\n',
                            '    strcpy(outaer6, OUT_DIR);\n',
                            '    strcat(outaer6, "albedo_NH3.dat");\n',
                            '    fim = fopen(outaer6, "r");\n',
                            '    for (j=1; j<=zbin; j++) {\n',
                            '        for (i=0; i < 324; i++) {fscanf(fim, "%lf", ccc+i);}\n',
                            '        for (i=' + str(iniz) + '; i<' + str(fine) + '; i++) {\n',
                            '            Interpolation( & wavelength[i], 1, & aNH3[j][i], lll, ccc, 324, 2);\n',
                            '        }\n',
                            '    }\n',
                            '    fclose(fim);\n',
                            '    bool amm_cld = true;\n']
        else:
            c_core_file += ['    bool amm_cld = false;\n']

                       # Geometric Albedo 9-point Gauss Quadruture
        c_core_file+= ['    double cmiu[9]={-0.9681602395076261,-0.8360311073266358,-0.6133714327005904,-0.3242534234038089,0.0,0.3242534234038089,0.6133714327005904,0.8360311073266358,0.9681602395076261};\n',
                       '    double wmiu[9]={0.0812743883615744,0.1806481606948574,0.2606106964029354,0.3123470770400029,0.3302393550012598,0.3123470770400029,0.2606106964029354,0.1806481606948574,0.0812743883615744};\n',
                       '    int NUMPOINTS=9;\n',
                       # 10-point Quadrature
                       # '    double cmiu[10]={-0.9739065285171717, -0.8650633666889845, -0.6794095682990244, -0.4333953941292472, -0.1488743389816312, 0.1488743389816312, 0.4333953941292472, 0.6794095682990244, 0.8650633666889845, 0.9739065285171717};\n',
                       # '    double wmiu[10]={0.0666713443086881, 0.1494513491505806, 0.2190863625159820, 0.2692667193099963, 0.2955242247147529, 0.2955242247147529, 0.2692667193099963, 0.2190863625159820, 0.1494513491505806, 0.0666713443086881};\n',
                       # '    int NUMPOINTS=10;\n',
                       # 11-point Quadrature
                       # '    double cmiu[11]={-0.9782286581460570, -0.8870625997680953, -0.7301520055740494, -0.5190961292068118, -0.2695431559523450, 0.0, 0.2695431559523450, 0.5190961292068118, 0.7301520055740494, 0.8870625997680953, 0.9782286581460570};\n',
                       # '    double wmiu[11]={0.0556685671161737, 0.1255803694649046, 0.1862902109277343, 0.2331937645919905, 0.2628045445102467, 0.2729250867779006, 0.2628045445102467, 0.2331937645919905, 0.1862902109277343, 0.1255803694649046, 0.0556685671161737};\n',
                       # '    int NUMPOINTS=11;\n',

                       '    double phase;\n',
                       '    phase = ' + str(self.param['phi']) + ';\n',  # Phase Angle, 0 zero geometric albedo
                       '    double lonfactor1, lonfactor2;\n',
                       '    double latfactor1, latfactor2;\n',
                       # '    lonfactor1 = (PI-phase)*0.5;\n',
                       # '    lonfactor2 = phase*0.5;\n',
                       '    lonfactor1 = PI*0.5;\n'
                       '    lonfactor2 = 0;\n'
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
                       '            Reflection(xx1, T, stdcross, qysum, cross, crosst, uvrfile, gmiu0, gmiu, phase, rout, ' + str(iniz) + ', ' + str(fine) + ', wtr_cld, amm_cld);\n',
                       '            for (k=' + str(iniz) + '; k < ' + str(fine) + '; k++) {\n',
                       '                gal[k] += wmiu[i] * wmiu[j] * rout[k] * gmiu * cos(lat[i]) * latfactor1 * lonfactor1 / PI;\n',
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
    

class RADIATIVE_TRANSFER_PYTHON:
    def __init__(self, param):
        self.param = copy.deepcopy(param)
        self.hazes_calc = param['hazes']
        self._python_core_cache = {}
        self.test = False

    def __surface_structure(self):
        test_verbose = bool(self.test)
        if test_verbose:
            _surface_timer_total = time.perf_counter()
            _surface_timer_block = _surface_timer_total

            def _print_timing(label):
                nonlocal _surface_timer_block
                now = time.perf_counter()
                print(f"__surface_structure [{label}]: {now - _surface_timer_block:.3f}s")
                _surface_timer_block = now

        wl_grid = np.asarray(self.param['opacw'], dtype=float).reshape(-1) * 1.0e6
        self.surf_alb = np.zeros(len(wl_grid))
        if self.param['surface_albedo_parameters'] == int(1):
            self.surf_alb += self.param['Ag']
        elif self.param['surface_albedo_parameters'] == int(3):
            left_mask = wl_grid < self.param['Ag_x1']
            self.surf_alb[left_mask] = self.param['Ag1']
            self.surf_alb[~left_mask] = self.param['Ag2']
        elif self.param['surface_albedo_parameters'] == int(5):
            left_mask = wl_grid < self.param['Ag_x1']
            middle_mask = (wl_grid >= self.param['Ag_x1']) & (wl_grid < self.param['Ag_x2'])
            right_mask = wl_grid >= self.param['Ag_x2']
            self.surf_alb[left_mask] = self.param['Ag1']
            self.surf_alb[middle_mask] = self.param['Ag2']
            self.surf_alb[right_mask] = self.param['Ag3']
        if test_verbose:
            _print_timing('surface_albedo_setup')

        self._python_core_cache['surface_albedo'] = self.surf_alb.copy()
        if test_verbose:
            _print_timing('surface_albedo_cache')
            print(f"__surface_structure [total]: {time.perf_counter() - _surface_timer_total:.3f}s")

    def __atmospheric_structure(self):
        test_verbose = bool(self.test)
        if test_verbose:
            _atmos_timer_total = time.perf_counter()
            _atmos_timer_block = _atmos_timer_total

            def _print_timing(label):
                nonlocal _atmos_timer_block
                now = time.perf_counter()
                print(f"__atmospheric_structure [{label}]: {now - _atmos_timer_block:.3f}s")
                _atmos_timer_block = now

        deltaP = 0.001 * 2.65495471  # assume super saturation to be 0.1% at 220 K

        g = self.param['gp'] + 0.0

        # Set up pressure grid
        P = self.param['P'] + 0.0  # in Pascal

        # Temperature profile
        T = self.param['T'] * np.ones(len(P)) # in K

        if self.param['fit_wtr_cld'] and 'H2O' in self.param['fit_molecules']:
            cloudden = 1.0e-36 * np.ones(len(P))
            for i in range(len(P) - 2, -1, -1):
                cloudden[i] = max(abs(self.param['vmr_H2O'][i] - self.param['vmr_H2O'][i + 1]) * 0.018 * P[i] / const.R.value / T[i], 1e-25)  # kg/m^3, g/L

            particlesize = 1.0e-36 * np.ones(len(P))
            if self.param['fit_p_size'] and self.param['p_size_type'] == 'constant':
                particlesize = self.param['p_size'] * np.ones(len(P))
            else:
                for i in range(len(P) - 2, -1, -1):
                    r0, _, r2, VP = particlesizef(g, T[i], P[i], self.param['mean_mol_weight'][i], self.param['mm']['H2O'], self.param['KE'], deltaP)
                    if self.param['fit_p_size'] and self.param['p_size_type'] == 'factor':
                        particlesize[i] = r2 * self.param['p_size']
                    else:
                        particlesize[i] = r2 + 0.0

        if self.param['fit_amm_cld'] and 'NH3' in self.param['fit_molecules']:
            cloudden_nh3 = 1.0e-36 * np.ones(len(P))
            for i in range(len(P) - 2, -1, -1):
                cloudden_nh3[i] = max(abs(self.param['vmr_NH3'][i] - self.param['vmr_NH3'][i + 1]) * 0.017 * P[i] / const.R.value / T[i], 1e-25)  # kg/m^3, g/L
            particlesize_nh3 = 1.0e-36 * np.ones(len(P))

            if self.param['fit_p_size'] and self.param['p_size_type'] == 'constant':
                particlesize_nh3 = self.param['p_size'] * np.ones(len(P))
            else:
                for i in range(len(P) - 2, -1, -1):
                    deltaP_nh3 = P[i] * abs(self.param['vmr_NH3'][i] - self.param['vmr_NH3'][i + 1])
                    r0, _, r2, VP = particlesizef(g, T[i], P[i], self.param['mean_mol_weight'][i], self.param['mm']['NH3'], self.param['KE'], deltaP_nh3)
                    if self.param['fit_p_size'] and self.param['p_size_type'] == 'factor':
                        particlesize_nh3[i] = r2 * self.param['p_size']
                    else:
                        particlesize_nh3[i] = r2 + 0.0

        if test_verbose:
            _print_timing('cloud_density_and_particle_size')

        # Calculate the height
        P = P[::-1]
        T = T[::-1]
        if self.param['fit_wtr_cld'] and 'H2O' in self.param['fit_molecules']:
            cloudden = cloudden[::-1]
            particlesize = particlesize[::-1]
        if self.param['fit_amm_cld'] and 'NH3' in self.param['fit_molecules']:
            cloudden_nh3 = cloudden_nh3[::-1]
            particlesize_nh3 = particlesize_nh3[::-1]
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
            idx_cloud_sets = []
            if self.param['fit_wtr_cld'] and 'H2O' in f:
                idx_cloud_sets.append(np.where(np.diff(f['H2O']) != 0.0)[0] + 1)
            if self.param['fit_amm_cld'] and 'NH3' in f:
                idx_cloud_sets.append(np.where(np.diff(f['NH3']) != 0.0)[0] + 1)
            if idx_cloud_sets:
                idx_cloud_layers = np.unique(np.concatenate(idx_cloud_sets))
            else:
                idx_cloud_layers = np.array([], dtype=int)
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

        if test_verbose:
            _print_timing('atmospheric_grid_setup')

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

        if self.param['fit_wtr_cld'] and 'H2O' in self.param['fit_molecules']:
            tck = interp1d(Z, np.log(cloudden))
            cloudden = np.exp(tck(zl))

            tck = interp1d(Z, np.log(particlesize))
            particlesize = np.exp(tck(zl))

        if self.param['fit_amm_cld'] and 'NH3' in self.param['fit_molecules']:
            tck = interp1d(Z, np.log(cloudden_nh3))
            cloudden_nh3 = np.exp(tck(zl))

            tck = interp1d(Z, np.log(particlesize_nh3))
            particlesize_nh3 = np.exp(tck(zl))

        self.species_to_num = self.__mol_species_number()
        NSP = 111
        xx_layers = np.zeros((len(zl), NSP + 1), dtype=float)
        for mol_name, values in n.items():
            std_num = self.species_to_num.get(mol_name)
            if std_num is None:
                continue
            if self.param['contribution'] and self.param['mol_contr'] != mol_name:
                continue
            xx_layers[:, std_num] = values

        if test_verbose:
            _print_timing('profile_interpolation')

        #    cloud output
        sig = 2
        n_cloud_wavelength = 324
        if self.param['fit_wtr_cld'] and 'H2O' in self.param['fit_molecules'] and self.param['wtr_cld_type'] == 'mixed' and self.param['PT_profile_type'] == 'parametric':
            ice_mask = tl < 273.15
            cro_h2o, alb_h2o, geo_h2o = self._compute_cloud_optics_from_tables(
                self.param['H2OL_r'],
                self.param['H2OL_c_liquid'],
                self.param['H2OL_a_liquid'],
                self.param['H2OL_g_liquid'],
                particlesize,
                cloudden * (~ice_mask),
                1.0,
                sig,
            )
            cro_h2o_ice, alb_h2o_ice, geo_h2o_ice = self._compute_cloud_optics_from_tables(
                self.param['H2OL_r'],
                self.param['H2OL_c_ice'],
                self.param['H2OL_a_ice'],
                self.param['H2OL_g_ice'],
                particlesize,
                cloudden * ice_mask,
                0.92,
                sig,
            )
            active_ice = ice_mask & (cloudden >= 1.0e-16)
            cro_h2o[active_ice] = cro_h2o_ice[active_ice]
            alb_h2o[active_ice] = alb_h2o_ice[active_ice]
            geo_h2o[active_ice] = geo_h2o_ice[active_ice]
        elif self.param['fit_wtr_cld'] and 'H2O' in self.param['fit_molecules']:
            water_material_density = 1.0 if self.param['wtr_cld_type'] == 'liquid' else 0.92
            cro_h2o, alb_h2o, geo_h2o = self._compute_cloud_optics_from_tables(
                self.param['H2OL_r'],
                self.param['H2OL_c'],
                self.param['H2OL_a'],
                self.param['H2OL_g'],
                particlesize,
                cloudden,
                water_material_density,
                sig,
            )
        else:
            cro_h2o = np.zeros((len(zl), n_cloud_wavelength), dtype=float)
            alb_h2o = np.ones((len(zl), n_cloud_wavelength), dtype=float)
            geo_h2o = np.zeros((len(zl), n_cloud_wavelength), dtype=float)

        if self.param['fit_amm_cld'] and 'NH3' in self.param['fit_molecules']:
            cro_nh3, alb_nh3, geo_nh3 = self._compute_cloud_optics_from_tables(
                self.param['NH3_r'],
                self.param['NH3_c'],
                self.param['NH3_a'],
                self.param['NH3_g'],
                particlesize_nh3,
                cloudden_nh3,
                0.87,
                sig,
            )

        if test_verbose:
            _print_timing('cloud_optics')

        layer_temperature = np.empty(len(tl) + 1, dtype=float)
        if len(tl) > 1:
            layer_temperature[1:-1] = 0.5 * (tl[:-1] + tl[1:])
            layer_temperature[0] = 1.5 * tl[0] - 0.5 * tl[1]
            layer_temperature[-1] = 1.5 * tl[-1] - 0.5 * tl[-2]
        else:
            layer_temperature[0] = tl[0]
            layer_temperature[-1] = tl[0]

        self._python_core_cache['profile'] = {
            'zl': np.asarray(zl, dtype=float),
            'z0': np.asarray(z0, dtype=float),
            'z1': np.asarray(z1, dtype=float),
            'tl': np.asarray(tl, dtype=float),
            'pl': np.asarray(pl, dtype=float),
            'mm': np.asarray(nden, dtype=float),
            'xx': np.asarray(xx_layers, dtype=float),
            'thickl': np.asarray((z1 - z0) * 1.0e5, dtype=float),
            'layer_temperature': layer_temperature,
        }
        self._python_core_cache['cloud_optics'] = {
            'wavelength_nm': 400.0 * np.power(1.0 + 1.0 / 200.0, np.arange(324, dtype=float)),
            'cross_h2o': np.asarray(cro_h2o, dtype=float),
            'albedo_h2o': np.asarray(alb_h2o, dtype=float),
            'g_h2o': np.asarray(geo_h2o, dtype=float),
        }
        if self.param['fit_amm_cld']:
            self._python_core_cache['cloud_optics'].update(
                {
                    'cross_nh3': np.asarray(cro_nh3, dtype=float),
                    'albedo_nh3': np.asarray(alb_nh3, dtype=float),
                    'g_nh3': np.asarray(geo_nh3, dtype=float),
                }
            )

        if test_verbose:
            _print_timing('atmosphere_cache')
            print(f"__atmospheric_structure [total]: {time.perf_counter() - _atmos_timer_total:.3f}s")
    
    def __mol_species_number(self):
        species_to_num = {
            'O': 1,
            'O3': 2,
            'H': 3,
            'OH': 4,
            'HO2': 5,
            'H2O2': 6,
            'H2O': 7,
            'N': 8,
            'NH3': 9,
            'NH2': 10,
            'N2O': 11,
            'NO': 12,
            'NO2': 13,
            'NO3': 14,
            'N2O5': 15,
            'HNO': 16,
            'HNO2': 17,
            'HNO3': 18,
            'C': 19,
            'CO': 20,
            'CH4': 21,
            'CH2O': 22,
            'CH2O2': 23,
            'CH4O': 24,
            'CH4O2': 25,
            'C2': 26,
            'C2H2': 27,
            'C2H3': 28,
            'C2H4': 29,
            'C2H5': 30,
            'C2H6': 31,
            'C2HO': 32,
            'C2H2O': 33,
            'C2H3O': 34,
            'C2H4O': 35,
            'C2H5O': 36,
            'HCN': 37,
            'CN': 38,
            'CNO': 39,
            'S': 40,
            'S2': 41,
            'SO': 42,
            'SO2': 43,
            'SO3': 44,
            'H2S': 45,
            'HS': 46,
            'HSO': 47,
            'HSO2': 48,
            'OCS': 49,
            'CS': 50,
            'CH3S': 51,
            'CO2': 52,
            'H2': 53,
            'O2': 54,
            'N2': 55,
            'O(1D)': 56,
            'NH': 57,
            'CH': 58,
            'CH2': 59,
            'CH3': 60,
            'CHO': 61,
            'CH3O': 62,
            'CHO2': 63,
            'C2H': 64,
            'CH3NO2': 65,
            'CH3NO3': 66,
            'C2H2N': 67,
            'HSO3': 68,
            'CH4S': 69,
            'HNO4': 70,
            'CH3O2': 71,
            'HCNO': 72,
            'H2SO4': 73,
            'SO21': 74,
            'SO23': 75,
            'S3': 76,
            'S4': 77,
            'H2SO4A': 78,
            'S8': 79,
            'CH21': 80,
            'C3H2': 81,
            'C3H3': 82,
            'C3H41': 83,
            'C3H42': 84,
            'C3H5': 85,
            'C3H6': 86,
            'C3H7': 87,
            'C3H8': 88,
            'C4H': 89,
            'C4H2': 90,
            'C4H3': 91,
            'C4H4': 92,
            'C4H5': 93,
            'C4H61': 94,
            'C4H62': 95,
            'C4H63': 96,
            'C4H8': 97,
            'C4H9': 98,
            'C4H10': 99,
            'C6H': 100,
            'C6H2': 101,
            'C6H3': 102,
            'C6H6': 103,
            'C8H2': 104,
            'CMHA': 105,
            'N2H2': 106,
            'N2H3': 107,
            'N2H4': 108,
            'CH5N': 109,
            'C2H5N': 110,
            'S8A': 111,
        }
        return species_to_num

    def _planck_nm(self, wavelength_m, temperature):
        exponent = np.clip(
            (const.h.value * const.c.value) / (wavelength_m * const.k_B.value * temperature),
            1.0e-12,
            700.0,
        )
        return (
            2.0
            * const.h.value
            * const.c.value ** 2
            / wavelength_m ** 5
            / np.expm1(exponent)
            * 1.0e-9
        )

    def _interpolate_table_rows(self, x_grid, table, targets):
        x_grid = np.asarray(x_grid, dtype=float).reshape(-1)
        values = np.asarray(table, dtype=float)
        targets = np.asarray(targets, dtype=float).reshape(-1)
        if values.ndim != 2:
            raise ValueError("Expected a 2D interpolation table.")

        if targets.size == 0:
            return np.zeros((0, values.shape[1]), dtype=float)
        if x_grid.size == 1:
            return np.repeat(values[:1], targets.size, axis=0)

        x_fit = np.clip(targets, x_grid[0], x_grid[-1])
        high = np.searchsorted(x_grid, x_fit, side='right')
        high = np.clip(high, 1, x_grid.size - 1)
        low = high - 1
        denom = x_grid[high] - x_grid[low]
        weight = np.divide(
            x_fit - x_grid[low],
            denom,
            out=np.zeros_like(x_fit),
            where=denom != 0.0,
        )
        low_values = values[low]
        high_values = values[high]
        return low_values + (high_values - low_values) * weight[:, np.newaxis]

    def _prepare_opacity_interpolation(self, temperature, pressure):
        temp_grid = np.asarray(self.param['opact'], dtype=float).reshape(-1)
        press_grid = np.asarray(self.param['opacp'], dtype=float).reshape(-1)
        n_layers = temperature.size

        if temp_grid.size == 1:
            t_low = np.zeros(n_layers, dtype=int)
            t_high = np.zeros(n_layers, dtype=int)
            t_weight = np.zeros(n_layers, dtype=float)
        else:
            t_fit = np.clip(temperature, temp_grid[0], temp_grid[-1])
            t_high = np.searchsorted(temp_grid, t_fit, side='right')
            t_high = np.clip(t_high, 1, temp_grid.size - 1)
            t_low = t_high - 1
            t_denom = temp_grid[t_high] - temp_grid[t_low]
            t_weight = np.divide(
                t_fit - temp_grid[t_low],
                t_denom,
                out=np.zeros_like(t_fit),
                where=t_denom != 0.0,
            )

        p_fit = np.minimum(pressure, press_grid[-1])
        valid_pressure = p_fit >= press_grid[0]
        if press_grid.size == 1:
            p_low = np.zeros(n_layers, dtype=int)
            p_high = np.zeros(n_layers, dtype=int)
            p_weight = np.zeros(n_layers, dtype=float)
        else:
            log_press_grid = np.log(press_grid)
            p_clipped = np.clip(p_fit, press_grid[0], press_grid[-1])
            p_high = np.searchsorted(press_grid, p_clipped, side='right')
            p_high = np.clip(p_high, 1, press_grid.size - 1)
            p_low = p_high - 1
            p_denom = log_press_grid[p_high] - log_press_grid[p_low]
            p_weight = np.divide(
                np.log(p_clipped) - log_press_grid[p_low],
                p_denom,
                out=np.zeros_like(p_clipped),
                where=p_denom != 0.0,
            )

        return {
            't_low': t_low,
            't_high': t_high,
            't_weight': t_weight,
            'p_low': p_low,
            'p_high': p_high,
            'p_weight': p_weight,
            'valid_pressure': valid_pressure,
        }

    def _evaluate_opacity_table(self, opacity_table, interp_state):
        table = np.asarray(opacity_table, dtype=float)
        n_layers = interp_state['t_low'].size
        n_wavelength = table.shape[-1]
        interpolated = np.zeros((n_layers, n_wavelength), dtype=float)
        valid_pressure = interp_state['valid_pressure']
        if not np.any(valid_pressure):
            return interpolated

        t_low = interp_state['t_low']
        t_high = interp_state['t_high']
        p_low = interp_state['p_low']
        p_high = interp_state['p_high']
        low_low = table[p_low, t_low]
        low_high = table[p_low, t_high]
        high_low = table[p_high, t_low]
        high_high = table[p_high, t_high]
        t_weight_2d = interp_state['t_weight'][:, np.newaxis]
        p_weight_2d = interp_state['p_weight'][:, np.newaxis]
        interp_low = low_low + (low_high - low_low) * t_weight_2d
        interp_high = high_low + (high_high - high_low) * t_weight_2d
        interpolated_valid = interp_low + (interp_high - interp_low) * p_weight_2d
        interpolated[valid_pressure] = np.maximum(interpolated_valid[valid_pressure], 0.0) * 1.0e4

        return interpolated

    def _evaluate_cia_tables(self, wavelength_nm, temperature):
        cia = self.param.get('cia', {})
        temp_grid = np.asarray(cia.get('temperature_grid', []), dtype=float)
        tables = cia.get('tables', {})
        n_layers = temperature.size
        n_wavelength = wavelength_nm.size
        output = {}

        for label in ['H2H2', 'H2He', 'H2H', 'N2H2', 'N2N2', 'CO2CO2', 'O2O2']:
            table = tables.get(label)
            if table is None or temp_grid.size == 0:
                output[label] = np.zeros((n_layers, n_wavelength), dtype=float)
                continue

            source_wavelength = np.asarray(table['wavelength'], dtype=float)
            values = np.asarray(table['values'], dtype=float)
            if source_wavelength.shape[0] != n_wavelength or not np.allclose(source_wavelength, wavelength_nm):
                resampled = np.empty((n_wavelength, values.shape[1]), dtype=float)
                for col in range(values.shape[1]):
                    resampled[:, col] = np.interp(
                        wavelength_nm,
                        source_wavelength,
                        values[:, col],
                        left=0.0,
                        right=0.0,
                    )
                values = resampled

            t_fit = np.clip(temperature, temp_grid[0], temp_grid[-1])
            if temp_grid.size == 1:
                output[label] = np.repeat(values[:, :1].T, n_layers, axis=0)
                continue

            t_high = np.searchsorted(temp_grid, t_fit, side='right')
            t_high = np.clip(t_high, 1, temp_grid.size - 1)
            t_low = t_high - 1
            log_temp_grid = np.log(temp_grid)
            t_denom = log_temp_grid[t_high] - log_temp_grid[t_low]
            t_weight = np.divide(
                np.log(t_fit) - log_temp_grid[t_low],
                t_denom,
                out=np.zeros_like(t_fit),
                where=t_denom != 0.0,
            )
            lower = values[:, t_low].T
            upper = values[:, t_high].T
            output[label] = lower + (upper - lower) * t_weight[:, np.newaxis]

        return output

    def _compute_rayleigh_terms(self, wavelength_nm, xx, mm):
        dens_standard = 101325.0 / const.k_B.value / 273.0 * 1.0e-6
        wavelength_um = wavelength_nm * 1.0e-3
        wavelength_cm = wavelength_nm * 1.0e-7

        def _rayleigh_from_refidx(refidx):
            ref_term = np.maximum(refidx, 1.0)
            return (
                1.061
                * 8.0
                * np.pi ** 3
                * (ref_term ** 2 - 1.0) ** 2
                / 3.0
                / wavelength_cm ** 4
                / dens_standard ** 2
            )

        co2_wl = np.clip(wavelength_nm, 480.0, 1800.0) * 1.0e-3
        co2_inv = 1.0 / co2_wl ** 2
        n2_inv = 1.0 / wavelength_um ** 2

        rayleigh_by_species = {
            7: _rayleigh_from_refidx(np.full_like(wavelength_nm, 1.000261)),
            21: _rayleigh_from_refidx(np.full_like(wavelength_nm, 1.000444)),
            52: _rayleigh_from_refidx(
                1.0
                + 1.0e-5
                * (
                    0.154489 / (0.0584738 - co2_inv)
                    + 8309.1927 / (210.92417 - co2_inv)
                    + 287.64190 / (60.122959 - co2_inv)
                )
            ),
            20: _rayleigh_from_refidx(np.full_like(wavelength_nm, 1.000338)),
            54: _rayleigh_from_refidx(1.0 + 1.181494e-4 + 9.708931e-3 / (75.4 - n2_inv)),
            2: _rayleigh_from_refidx(np.full_like(wavelength_nm, 1.00052)),
            11: _rayleigh_from_refidx(np.full_like(wavelength_nm, 1.000516)),
            55: _rayleigh_from_refidx(1.0 + 6.8552e-5 + 3.243157e-2 / (144.0 - n2_inv)),
            53: 8.14e-13 * (wavelength_nm * 10.0) ** -4
            + 1.28e-6 * (wavelength_nm * 10.0) ** -6
            + 1.61 * (wavelength_nm * 10.0) ** -8,
        }

        species_order = np.array([7, 21, 52, 20, 54, 2, 11, 55, 53], dtype=int)
        mixing_ratio = np.divide(
            xx[:, species_order],
            mm[:, np.newaxis],
            out=np.zeros((xx.shape[0], species_order.size), dtype=float),
            where=mm[:, np.newaxis] > 0.0,
        )
        coefficients = np.vstack([rayleigh_by_species[idx] for idx in species_order])
        combined = mixing_ratio @ coefficients

        if not self.param['contribution']:
            denominator = np.sum(mixing_ratio, axis=1)
            crossr = np.zeros_like(combined)
            valid = denominator > 0.0
            crossr[valid] = combined[valid] / denominator[valid, np.newaxis]
            return crossr
        if self.param['mol_contr'] is not None:
            return combined
        return np.zeros_like(combined)

    def _compute_cloud_optics_from_tables(
        self,
        radius_grid,
        cross_table,
        albedo_table,
        g_table,
        particle_radius,
        cloud_density,
        material_density,
        sig,
    ):
        n_layers = particle_radius.size
        n_wavelength = np.asarray(cross_table, dtype=float).shape[1]
        cross = np.zeros((n_layers, n_wavelength), dtype=float)
        albedo = np.ones((n_layers, n_wavelength), dtype=float)
        asymmetry = np.zeros((n_layers, n_wavelength), dtype=float)

        active = cloud_density >= 1.0e-16
        if not np.any(active):
            return cross, albedo, asymmetry

        log_radius_grid = np.log10(np.asarray(radius_grid, dtype=float))
        sig_log_sq = np.log(sig) ** 2
        active_radius = particle_radius[active]
        target_radius = np.log10(np.clip(active_radius * np.exp(-sig_log_sq), 0.01, 100.0))
        vp = (
            4.0
            * math.pi
            / 3.0
            * ((active_radius * 1.0e-6 * np.exp(0.5 * sig_log_sq)) ** 3.0)
            * 1.0e6
            * material_density
        )

        log_cross = self._interpolate_table_rows(log_radius_grid, np.log10(cross_table), target_radius)
        albedo_active = self._interpolate_table_rows(log_radius_grid, albedo_table, target_radius)
        asymmetry_active = self._interpolate_table_rows(log_radius_grid, g_table, target_radius)

        cross[active] = cloud_density[active][:, np.newaxis] / vp[:, np.newaxis] * 1.0e-3 * np.power(10.0, log_cross)
        albedo[active] = albedo_active
        asymmetry[active] = asymmetry_active
        return cross, albedo, asymmetry

    def _build_optical_properties(self, wavelength_nm, tl, pl, mm, xx, thickl, cloud_optics):
        n_layers = tl.size
        n_wavelength = wavelength_nm.size
        wa = np.zeros((n_layers, n_wavelength), dtype=float)
        ws = np.zeros((n_layers, n_wavelength), dtype=float)
        g_numerator = np.zeros((n_layers, n_wavelength), dtype=float)
        opacity_interp = self._prepare_opacity_interpolation(tl, pl)

        clipped_cross_temperature = np.clip(tl, 200.0, 300.0)
        for std_num, table in self.param.get('photolysis_tables', {}).items():
            if std_num >= xx.shape[1]:
                continue
            number_density = xx[:, std_num]
            if not np.any(number_density):
                continue
            data = np.asarray(table['data'], dtype=float)
            cross = np.interp(wavelength_nm, data[:, 0], data[:, 1], left=0.0, right=0.0)
            cross_t = np.interp(wavelength_nm, data[:, 0], data[:, 2], left=0.0, right=0.0)
            wa += number_density[:, np.newaxis] * (
                cross[np.newaxis, :] + (clipped_cross_temperature[:, np.newaxis] - 295.0) * cross_t[np.newaxis, :]
            )

        active_molecules = list(self.param.get('fit_molecules', []))
        gas_fill = self.param.get('gas_fill')
        if gas_fill is not None and gas_fill not in active_molecules:
            active_molecules.append(gas_fill)
        for mol_name in active_molecules:
            param_key = 'opac' + mol_name.lower()
            if param_key not in self.param:
                continue
            std_num = self.species_to_num.get(mol_name)
            if std_num is None:
                continue
            number_density = xx[:, std_num]
            if not np.any(number_density):
                continue
            sigma = self._evaluate_opacity_table(self.param[param_key], opacity_interp)
            wa += sigma * number_density[:, np.newaxis]

        cia_terms = self._evaluate_cia_tables(wavelength_nm, tl)
        helium_number = np.maximum(mm - np.sum(xx[:, 1:], axis=1), 0.0)
        wa += cia_terms['H2H2'] * xx[:, 53][:, np.newaxis] * xx[:, 53][:, np.newaxis]
        wa += cia_terms['H2H'] * xx[:, 53][:, np.newaxis] * xx[:, 3][:, np.newaxis]
        wa += cia_terms['H2He'] * xx[:, 53][:, np.newaxis] * helium_number[:, np.newaxis]
        wa += cia_terms['N2H2'] * xx[:, 53][:, np.newaxis] * xx[:, 55][:, np.newaxis]
        wa += cia_terms['N2N2'] * xx[:, 55][:, np.newaxis] * xx[:, 55][:, np.newaxis]
        wa += cia_terms['CO2CO2'] * xx[:, 52][:, np.newaxis] * xx[:, 52][:, np.newaxis]
        wa += cia_terms['O2O2'] * xx[:, 54][:, np.newaxis] * xx[:, 54][:, np.newaxis]

        cloud_wavelength = np.asarray(cloud_optics['wavelength_nm'], dtype=float)
        def _resample_cloud_term(values):
            resampled = np.empty((n_layers, n_wavelength), dtype=float)
            for layer_idx in range(n_layers):
                resampled[layer_idx] = np.interp(
                    wavelength_nm,
                    cloud_wavelength,
                    values[layer_idx],
                    left=values[layer_idx, 0],
                    right=values[layer_idx, -1],
                )
            return resampled

        cross_h2o = _resample_cloud_term(np.asarray(cloud_optics['cross_h2o'], dtype=float))
        albedo_h2o = _resample_cloud_term(np.asarray(cloud_optics['albedo_h2o'], dtype=float))
        g_h2o = _resample_cloud_term(np.asarray(cloud_optics['g_h2o'], dtype=float))

        wa += cross_h2o * (1.0 - albedo_h2o)
        ws += cross_h2o * albedo_h2o
        g_numerator += cross_h2o * albedo_h2o * g_h2o

        if 'cross_nh3' in cloud_optics:
            cross_nh3 = _resample_cloud_term(np.asarray(cloud_optics['cross_nh3'], dtype=float))
            albedo_nh3 = _resample_cloud_term(np.asarray(cloud_optics['albedo_nh3'], dtype=float))
            g_nh3 = _resample_cloud_term(np.asarray(cloud_optics['g_nh3'], dtype=float))
            wa += cross_nh3 * (1.0 - albedo_nh3)
            ws += cross_nh3 * albedo_nh3
            g_numerator += cross_nh3 * albedo_nh3 * g_nh3

        crossr = self._compute_rayleigh_terms(wavelength_nm, xx, mm)
        rayleigh_scattering = crossr * mm[:, np.newaxis]
        ws += rayleigh_scattering

        w = np.zeros_like(ws)
        g = np.zeros_like(ws)
        rr = np.ones_like(ws)
        valid_scattering = ws > 0.0
        total_extinction = wa + ws
        w[valid_scattering] = ws[valid_scattering] / total_extinction[valid_scattering]
        g[valid_scattering] = g_numerator[valid_scattering] / ws[valid_scattering]
        rr[valid_scattering] = rayleigh_scattering[valid_scattering] / ws[valid_scattering]
        w = np.clip(w, 1.0e-13, 1.0 - 1.0e-13)

        tau = total_extinction * thickl[:, np.newaxis]
        delta_scale = 1.0 - w * g * g
        tau = tau * delta_scale
        w = ((1.0 - g * g) * w) / delta_scale
        g = g / (1.0 + g)
        tc = np.vstack([np.zeros((1, n_wavelength), dtype=float), np.cumsum(tau, axis=0)])

        return {
            'tau': tau,
            'w': w,
            'g': g,
            'rr': rr,
            'tc': tc,
        }

    def _prepare_reflection_groups(self, optical, solar, wavelength_m, surface_albedo, planck_boundary, phase):
        tau_all = optical['tau']
        w_all = optical['w']
        g_all = optical['g']
        rr_all = optical['rr']
        tc_all = optical['tc']
        n_layers, n_wavelength = tau_all.shape
        positive_solar = solar > 0.0
        if not np.any(positive_solar):
            return {'n_wavelength': n_wavelength, 'groups': []}

        tau_limit = tc_all >= 1000.0
        ntau_by_wave = tau_limit.argmax(axis=0)
        ntau_by_wave[~tau_limit.any(axis=0)] = n_layers
        ntau_by_wave = np.clip(ntau_by_wave, 1, n_layers)
        cos_scattering = math.cos(np.pi - phase)
        groups = []

        for ntau in np.unique(ntau_by_wave[positive_solar]):
            wave_idx = np.where((ntau_by_wave == ntau) & positive_solar)[0]
            tau1 = tau_all[:ntau, wave_idx]
            w1 = w_all[:ntau, wave_idx]
            g1 = g_all[:ntau, wave_idx]
            rr1 = rr_all[:ntau, wave_idx]
            tc1 = tc_all[:ntau + 1, wave_idx]
            planck1 = planck_boundary[:ntau + 1, wave_idx]
            solar_i = solar[wave_idx]
            surfaceref = np.where(wavelength_m[wave_idx] * 1.0e9 > 3000.0, 0.0, surface_albedo[wave_idx])

            gamma1 = (7.0 - (4.0 + 3.0 * g1) * w1) / 4.0
            gamma2 = -((1.0 - (4.0 - 3.0 * g1) * w1) / 4.0)
            lam = np.sqrt(np.maximum(gamma1 * gamma1 - gamma2 * gamma2, 0.0))
            gamma = gamma2 / (gamma1 + lam)
            exp_term = np.exp(-lam * tau1)
            e1 = 1.0 + gamma * exp_term
            e2 = 1.0 - gamma * exp_term
            e3 = gamma + exp_term
            e4 = gamma - exp_term

            phase_function = rr1 * 0.75 * (cos_scattering * cos_scattering + 1.0)
            hg_term = (
                (1.0 - g1 * g1 / 4.0)
                * (1.0 - g1 * g1)
                / np.power(1.0 + g1 * g1 - 2.0 * g1 * cos_scattering, 1.5)
                + g1 * g1
                / 4.0
                * (1.0 - g1 * g1 / 4.0)
                / np.power(1.0 + g1 * g1 / 4.0 + g1 * cos_scattering, 1.5)
            )
            phase_function += (1.0 - rr1) * hg_term

            groups.append(
                {
                    'ntau': int(ntau),
                    'nvector': 2 * int(ntau),
                    'wave_idx': wave_idx,
                    'tau': tau1,
                    'w': w1,
                    'g': g1,
                    'gamma1': gamma1,
                    'gamma2': gamma2,
                    'lam': lam,
                    'gamma': gamma,
                    'e1': e1,
                    'e2': e2,
                    'e3': e3,
                    'e4': e4,
                    'tc': tc1,
                    'solar': solar_i,
                    'surface_reflectance': surfaceref,
                    'phase_function': phase_function,
                    'aa1': 2.0 * np.pi * planck1[:-1],
                    'planck_source': 2.0 * np.pi * 0.5 * planck1[:-1],
                    'bottom_planck': planck1[-1],
                }
            )

        return {'n_wavelength': n_wavelength, 'groups': groups}

    def _solve_reflection_angle(self, reflection_data, miu0, mium):
        rout = np.zeros(reflection_data['n_wavelength'], dtype=float)
        miu0_diffuse = min(miu0, 1.0)

        for group in reflection_data['groups']:
            ntau = group['ntau']
            wave_idx = group['wave_idx']
            tau1 = group['tau']
            w1 = group['w']
            g1 = group['g']
            gamma1 = group['gamma1']
            gamma2 = group['gamma2']
            lam = group['lam']
            gamma = group['gamma']
            e1 = group['e1']
            e2 = group['e2']
            e3 = group['e3']
            e4 = group['e4']
            tc1 = group['tc']
            solar_i = group['solar']
            surfaceref = group['surface_reflectance']

            gamma3 = (2.0 - 3.0 * g1 * miu0_diffuse) / 4.0
            gamma4 = 1.0 - gamma3

            cp0 = group['planck_source'].copy()
            cp1 = cp0.copy()
            cm0 = cp0.copy()
            cm1 = cp0.copy()

            if miu0_diffuse > 0.0:
                denom = lam * lam - 1.0 / (miu0_diffuse * miu0_diffuse)
                exp_top = np.exp(-tc1[:-1] / miu0_diffuse)
                exp_bottom = np.exp(-tc1[1:] / miu0_diffuse)
                solar_prefactor = solar_i[np.newaxis, :] * w1
                cp0 += solar_prefactor * exp_top * (
                    ((gamma1 - 1.0 / miu0_diffuse) * gamma3 + gamma2 * gamma4) / denom
                )
                cp1 += solar_prefactor * exp_bottom * (
                    ((gamma1 - 1.0 / miu0_diffuse) * gamma3 + gamma2 * gamma4) / denom
                )
                cm0 += solar_prefactor * exp_top * (
                    ((gamma1 + 1.0 / miu0_diffuse) * gamma4 + gamma2 * gamma3) / denom
                )
                cm1 += solar_prefactor * exp_bottom * (
                    ((gamma1 + 1.0 / miu0_diffuse) * gamma4 + gamma2 * gamma3) / denom
                )

            nvector = group['nvector']
            a = np.zeros((nvector, wave_idx.size), dtype=float)
            b = np.zeros((nvector, wave_idx.size), dtype=float)
            d = np.zeros((nvector, wave_idx.size), dtype=float)
            e = np.zeros((nvector, wave_idx.size), dtype=float)

            a[0] = 0.0
            b[0] = e1[0]
            d[0] = -e2[0]
            e[0] = -cm0[0]

            for layer_idx in range(ntau - 1):
                even_row = 2 * layer_idx + 1
                odd_row = 2 * layer_idx + 2
                a[odd_row] = e2[layer_idx] * e3[layer_idx] - e4[layer_idx] * e1[layer_idx]
                b[odd_row] = e1[layer_idx] * e1[layer_idx + 1] - e3[layer_idx] * e3[layer_idx + 1]
                d[odd_row] = e3[layer_idx] * e4[layer_idx + 1] - e1[layer_idx] * e2[layer_idx + 1]
                e[odd_row] = e3[layer_idx] * (cp0[layer_idx + 1] - cp1[layer_idx]) - e1[layer_idx] * (
                    cm0[layer_idx + 1] - cm1[layer_idx]
                )

                a[even_row] = e2[layer_idx + 1] * e1[layer_idx] - e3[layer_idx] * e4[layer_idx + 1]
                b[even_row] = e2[layer_idx] * e2[layer_idx + 1] - e4[layer_idx] * e4[layer_idx + 1]
                d[even_row] = e1[layer_idx + 1] * e4[layer_idx + 1] - e2[layer_idx + 1] * e3[layer_idx + 1]
                e[even_row] = e2[layer_idx + 1] * (cp0[layer_idx + 1] - cp1[layer_idx]) - e4[layer_idx + 1] * (
                    cm0[layer_idx + 1] - cm1[layer_idx]
                )

            a[-1] = e1[-1] - surfaceref * e3[-1]
            b[-1] = e2[-1] - surfaceref * e4[-1]
            d[-1] = 0.0
            e[-1] = -cp1[-1] + surfaceref * cm1[-1] + (1.0 - surfaceref) * np.pi * group['bottom_planck']
            if miu0_diffuse > 0.0:
                e[-1] += surfaceref * miu0_diffuse * solar_i * np.exp(-tc1[-1] / miu0_diffuse)

            as_ = np.zeros_like(a)
            ds_ = np.zeros_like(a)
            y = np.zeros_like(a)
            as_[-1] = a[-1] / b[-1]
            ds_[-1] = e[-1] / b[-1]
            for row_idx in range(nvector - 2, -1, -1):
                factor = 1.0 / (b[row_idx] - d[row_idx] * as_[row_idx + 1])
                as_[row_idx] = a[row_idx] * factor
                ds_[row_idx] = (e[row_idx] - d[row_idx] * ds_[row_idx + 1]) * factor
            y[0] = ds_[0]
            for row_idx in range(1, nvector):
                y[row_idx] = ds_[row_idx] - as_[row_idx] * y[row_idx - 1]
            y1 = y[0::2]
            y2 = y[1::2]

            aa1 = group['aa1']
            aa2 = np.zeros_like(aa1)
            aa3 = np.zeros_like(aa1)
            if miu0 > 0.0:
                denom = lam * lam - 1.0 / (miu0 * miu0)
                aa3 = (
                    w1
                    / 2.0
                    * solar_i[np.newaxis, :]
                    * (
                        group['phase_function']
                        + (2.0 + 3.0 * g1 * mium)
                        * w1
                        * ((gamma1 - 1.0 / miu0) * gamma3 + gamma2 * gamma4)
                        / denom
                        + (2.0 - 3.0 * g1 * mium)
                        * w1
                        * ((gamma1 + 1.0 / miu0) * gamma4 + gamma2 * gamma3)
                        / denom
                    )
                    * np.exp(-tc1[:-1] / miu0)
                )
            aa4r = (y1 + y2) * w1 / 2.0 * (2.0 + 3.0 * g1 * mium + gamma * (2.0 - 3.0 * g1 * mium))
            aa5 = (y1 - y2) * w1 / 2.0 * (((2.0 + 3.0 * g1 * mium) * gamma) + 2.0 - 3.0 * g1 * mium)

            same_mu = abs(mium - miu0) == 0.0
            rid = np.zeros(wave_idx.size, dtype=float)
            for layer_idx in range(ntau):
                exp_m = np.exp(-tau1[layer_idx] / mium)
                if same_mu:
                    rid = (
                        rid * exp_m
                        + aa1[layer_idx] * (1.0 - exp_m)
                        + aa2[layer_idx] * (tau1[layer_idx] - mium + mium * exp_m)
                        + aa3[layer_idx] * tau1[layer_idx] / mium * exp_m
                        + aa4r[layer_idx]
                        / (1.0 + mium * lam[layer_idx])
                        * (1.0 - np.exp(-tau1[layer_idx] / mium - lam[layer_idx] * tau1[layer_idx]))
                        + aa5[layer_idx]
                        / (1.0 - mium * lam[layer_idx])
                        * (np.exp(-lam[layer_idx] * tau1[layer_idx]) - exp_m)
                    )
                elif miu0 < 0.0:
                    rid = (
                        rid * exp_m
                        + aa1[layer_idx] * (1.0 - exp_m)
                        + aa2[layer_idx] * (tau1[layer_idx] - mium + mium * exp_m)
                        + aa4r[layer_idx]
                        / (1.0 + mium * lam[layer_idx])
                        * (1.0 - np.exp(-tau1[layer_idx] / mium - lam[layer_idx] * tau1[layer_idx]))
                        + aa5[layer_idx]
                        / (1.0 - mium * lam[layer_idx])
                        * (np.exp(-lam[layer_idx] * tau1[layer_idx]) - exp_m)
                    )
                else:
                    rid = (
                        rid * exp_m
                        + aa1[layer_idx] * (1.0 - exp_m)
                        + aa2[layer_idx] * (tau1[layer_idx] - mium + mium * exp_m)
                        + aa3[layer_idx]
                        * miu0
                        / (miu0 - mium)
                        * (np.exp(-tau1[layer_idx] / miu0) - exp_m)
                        + aa4r[layer_idx]
                        / (1.0 + mium * lam[layer_idx])
                        * (1.0 - np.exp(-tau1[layer_idx] / mium - lam[layer_idx] * tau1[layer_idx]))
                        + aa5[layer_idx]
                        / (1.0 - mium * lam[layer_idx])
                        * (np.exp(-lam[layer_idx] * tau1[layer_idx]) - exp_m)
                    )

            riu = rid * surfaceref + (1.0 - surfaceref) * 2.0 * np.pi * group['bottom_planck']
            if miu0_diffuse > 0.0:
                riu += surfaceref * 2.0 * miu0_diffuse * solar_i * np.exp(-tc1[-1] / miu0_diffuse)
                for layer_idx in range(ntau - 1, -1, -1):
                    exp_m = np.exp(-tau1[layer_idx] / mium)
                    riu = (
                        riu * exp_m
                        + aa1[layer_idx] * (1.0 - exp_m)
                        + aa2[layer_idx] * (mium - (tau1[layer_idx] + mium) * exp_m)
                        + aa3[layer_idx]
                        * miu0
                        / (miu0 + mium)
                        * (1.0 - np.exp(-tau1[layer_idx] / miu0 - tau1[layer_idx] / mium))
                        + aa4r[layer_idx]
                        / (1.0 - mium * lam[layer_idx])
                        * (np.exp(-lam[layer_idx] * tau1[layer_idx]) - exp_m)
                        + aa5[layer_idx]
                        / (1.0 + mium * lam[layer_idx])
                        * (1.0 - np.exp(-tau1[layer_idx] / mium - lam[layer_idx] * tau1[layer_idx]))
                    )
            else:
                for layer_idx in range(ntau - 1, -1, -1):
                    exp_m = np.exp(-tau1[layer_idx] / mium)
                    riu = (
                        riu * exp_m
                        + aa1[layer_idx] * (1.0 - exp_m)
                        + aa2[layer_idx] * (mium - (tau1[layer_idx] + mium) * exp_m)
                        + aa4r[layer_idx]
                        / (1.0 - mium * lam[layer_idx])
                        * (np.exp(-lam[layer_idx] * tau1[layer_idx]) - exp_m)
                        + aa5[layer_idx]
                        / (1.0 + mium * lam[layer_idx])
                        * (1.0 - np.exp(-tau1[layer_idx] / mium - lam[layer_idx] * tau1[layer_idx]))
                    )

            values = riu / solar_i / 2.0
            values[~np.isfinite(values)] = 0.0
            rout[wave_idx] = values

        return rout

    def __core_function(self):
        # Architectural notes for the future Python radiative-transfer core:
        # keep the raw opacity tables in self.param and build any interpolator
        # objects here, close to the runtime logic that consumes them.
        #
        # Do not fuse everything into one giant sigma(T, P, wl) interpolator.
        # That makes it harder to mirror the C behavior and harder to debug.
        #
        # Preferred split:
        # - OpacityTable: owns temp_grid, press_grid, wl_grid, values
        # - OpacityEvaluator: returns a spectrum on the native opacity
        #   wavelength grid for a given T and P
        # - WavelengthResampler: maps that native-grid spectrum onto the
        #   radiative-transfer or instrument wavelength grid
        #
        # This also keeps the interpolation semantics explicit:
        # - interpolate in T and log(P), matching the C implementation
        # - keep the raw arrays in param
        # - create interpolators in __core_function
        # - CIA tables are resampled onto the same working wavelength grid as
        #   the cross sections, so both should be consumed on a shared wl grid
        #
        # TODO: when wiring the Python opacity evaluator, verify whether the
        # loaded cross sections should be converted from m^2 to cm^2 here to
        # match forward_mod/readcross.c exactly.
        profile = self._python_core_cache.get('profile')
        cloud_optics = self._python_core_cache.get('cloud_optics')
        if profile is None or cloud_optics is None:
            raise RuntimeError("Atmospheric structure must be computed before the Python core runs.")

        wavelength_m = np.asarray(self.param['opacw'], dtype=float).reshape(-1)
        wavelength_nm = wavelength_m * 1.0e9
        solar_data = np.asarray(self.param['solar_data'], dtype=float)
        solar = np.interp(wavelength_nm, solar_data[:, 0], solar_data[:, 1], left=0.0, right=0.0)
        solar = solar / (self.param['equivalent_a'] ** 2)
        solar_tail_start = 0
        while solar_tail_start < solar.size and (solar[solar_tail_start] > 0.0 or wavelength_nm[solar_tail_start] < 9990.0):
            solar_tail_start += 1
        if 0 < solar_tail_start < solar.size:
            solar[solar_tail_start:] = solar[solar_tail_start - 1] * (
                wavelength_nm[solar_tail_start - 1] / wavelength_nm[solar_tail_start:]
            ) ** 4

        tl = np.asarray(profile['tl'], dtype=float)[::-1]
        pl = np.asarray(profile['pl'], dtype=float)[::-1]
        mm = np.asarray(profile['mm'], dtype=float)[::-1]
        xx = np.asarray(profile['xx'], dtype=float)[::-1]
        thickl = np.asarray(profile['thickl'], dtype=float)[::-1]
        layer_temperature = np.asarray(profile['layer_temperature'], dtype=float)[::-1]
        cloud_top = {
            'wavelength_nm': np.asarray(cloud_optics['wavelength_nm'], dtype=float),
            'cross_h2o': np.asarray(cloud_optics['cross_h2o'], dtype=float)[::-1],
            'albedo_h2o': np.asarray(cloud_optics['albedo_h2o'], dtype=float)[::-1],
            'g_h2o': np.asarray(cloud_optics['g_h2o'], dtype=float)[::-1],
        }
        if 'cross_nh3' in cloud_optics:
            cloud_top.update(
                {
                    'cross_nh3': np.asarray(cloud_optics['cross_nh3'], dtype=float)[::-1],
                    'albedo_nh3': np.asarray(cloud_optics['albedo_nh3'], dtype=float)[::-1],
                    'g_nh3': np.asarray(cloud_optics['g_nh3'], dtype=float)[::-1],
                }
            )

        optical = self._build_optical_properties(wavelength_nm, tl, pl, mm, xx, thickl, cloud_top)
        planck_boundary = self._planck_nm(wavelength_m[np.newaxis, :], layer_temperature[:, np.newaxis])
        surface_albedo = np.asarray(self._python_core_cache['surface_albedo'], dtype=float)

        cmiu = np.array(
            [
                -0.9681602395076261,
                -0.8360311073266358,
                -0.6133714327005904,
                -0.3242534234038089,
                0.0,
                0.3242534234038089,
                0.6133714327005904,
                0.8360311073266358,
                0.9681602395076261,
            ],
            dtype=float,
        )
        wmiu = np.array(
            [
                0.0812743883615744,
                0.1806481606948574,
                0.2606106964029354,
                0.3123470770400029,
                0.3302393550012598,
                0.3123470770400029,
                0.2606106964029354,
                0.1806481606948574,
                0.0812743883615744,
            ],
            dtype=float,
        )
        lat = 0.5 * np.pi * cmiu
        lon = 0.5 * np.pi * cmiu
        phase = float(self.param['phi'])
        reflection_data = self._prepare_reflection_groups(
            optical,
            solar,
            wavelength_m,
            surface_albedo,
            planck_boundary,
            phase,
        )

        geometric_albedo = np.zeros_like(wavelength_nm, dtype=float)
        for lat_idx in range(9):
            cos_lat = math.cos(lat[lat_idx])
            for lon_idx in range(9):
                gmiu0 = cos_lat * math.cos(lon[lon_idx] - phase)
                gmiu = cos_lat * math.cos(lon[lon_idx])
                if abs(gmiu0 - gmiu) < 1.0e-7:
                    gmiu = gmiu0 + 1.0e-7
                rout = self._solve_reflection_angle(reflection_data, gmiu0, gmiu)
                geometric_albedo += (
                    wmiu[lat_idx]
                    * wmiu[lon_idx]
                    * rout
                    * gmiu
                    * cos_lat
                    * (0.5 * np.pi)
                    * (0.5 * np.pi)
                    / np.pi
                )

        return wavelength_nm, geometric_albedo

    def run_forward(self):
        self.__atmospheric_structure()
        self.__surface_structure()
        alb_wl, alb = self.__core_function()

        alb_wl *= 10. ** (-3.)

        return alb_wl, alb


def forward(parameters_dictionary, evaluation=None, phi=None, n_obs=None, retrieval_mode=True, core_number=None, albedo_calc=False, fp_over_fs=False, canc_metadata=False):
    param = copy.deepcopy(parameters_dictionary)

    if evaluation is not None:
        if param['fit_p0'] or param['gas_par_space'] == 'partial_pressure':
            param['P0'] = evaluation['P0']
        if param['fit_wtr_cld'] and param['PT_profile_type'] == 'isothermal':
            param['Pw_top'] = evaluation['pH2O']
            param['cldw_depth'] = evaluation['dH2O']
            param['CR_H2O'] = evaluation['crH2O']
        if param['fit_amm_cld'] and param['PT_profile_type'] == 'isothermal':
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
            if param['PT_profile_type'] == 'isothermal':
                param['Tp'] = evaluation['Tp']
            elif param['PT_profile_type'] == 'parametric':
                param['kappa_th'] = evaluation['kappa_th']
                param['gamma'] = evaluation['gamma']
                param['beta'] = evaluation['beta']
                if param['fit_Tint']:
                    param['Tint'] = evaluation['Tint']
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
    if param['PT_profile_type'] == 'isothermal' or param['PT_profile_type'] == 'parametric':
        param['P'] = 10. ** np.arange(0.0, np.log10(param['P0']) + 0.01, step=0.01)
        param['T'] = temp_profile(param)
    else:
        param['P'] = param['Pp'] + 0.0
        param['T'] = param['Tp'] + 0.0
    if param['fit_amm_cld']:
        param['vmr_NH3'] = cloud_pos(param, condensed_gas='NH3')
        param = adjust_VMR(param, all_gases=param['adjust_VMR_gases'], condensed_gas='NH3')
    if param['fit_wtr_cld']:
        param['vmr_H2O'] = cloud_pos(param, condensed_gas='H2O')
        param = adjust_VMR(param, all_gases=param['adjust_VMR_gases'], condensed_gas='H2O')
    if not param['fit_wtr_cld'] and not param['fit_amm_cld']:
        param = adjust_VMR(param, all_gases=param['adjust_VMR_gases'], condensed_gas=None)
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
        if param['physics_model_code_language'] == 'Python':
            mod = RADIATIVE_TRANSFER_PYTHON(param)
        else:
            mod = RADIATIVE_TRANSFER_C(param, retrieval=retrieval_mode, canc_metadata=canc_metadata)
        
    # elif param['physics_model'] == 'dataset':
    #     mod = FORWARD_DATASET(param, dataset_dir=param['dataset_dir'])
    # elif param['physics_model'] == 'AI_model':
    #     mod = FORWARD_AI(param)
    else:
        raise ValueError('Unknown physics_model: ' + str(param['physics_model']))
    
    alb_wl, alb = mod.run_forward()
    wl, model = model_finalizzation(param, alb_wl, alb, planet_albedo=albedo_calc, fp_over_fs=fp_over_fs, n_obs=n_obs)

    if retrieval_mode:
        return model
    else:
        return wl, model
