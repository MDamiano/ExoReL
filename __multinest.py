from .__basics import *
from .__utils import *
from .__forward import *
from .__plotting import *
from . import __version__

# trying to initiate MPI parallelization
try:
    from mpi4py import MPI

    MPIrank = MPI.COMM_WORLD.Get_rank()
    MPIsize = MPI.COMM_WORLD.Get_size()
    MPIimport = True
except ImportError:
    MPIimport = False

if MPIimport:
    if MPIrank == 1:
        print('MPI enabled. Running on ' + str(MPIsize) + ' cores')
else:
    print('MPI disabled')

# checking for multinest library
try:
    import pymultinest

    multinest_import = True
except:
    multinest_import = False

if multinest_import:
    if MPIrank == 1:
        from pymultinest.run import lib_mpi

        print('MultiNest library: "' + str(lib_mpi) + '" correctly loaded.')
else:
    raise ImportError ('ERRORS OCCURRED - Check your MultiNest installation.')


class MULTINEST:
    def __init__(self, par):
        self.param = copy.deepcopy(par)
        self.param = par_and_calc(self.param)
        self.param = load_input_spectrum(self.param)
        if not self.param['albedo_calc'] and not self.param['fp_over_fs']:
            self.param = take_star_spectrum(self.param)
        self.param = pre_load_variables(self.param)
        self.param = ranges(self.param)

    def run_retrieval(self):
        if MPIimport and MPIrank == 0:
            print(f"Running ExoReL – version {__version__}")
            # check if the run is done, in case clean c meta files
            if not os.path.isfile(self.param['out_dir'] + self.param['name_p'] + '_params.json'):
                clean_c_files(self.param['pkg_dir'])
            if self.param['rocky']:
                print('Using ExoReL-R for small planets')
                if self.param['gas_par_space'] == 'clr' or self.param['gas_par_space'] == 'centered_log_ratio':
                    print('Using modified priors: ' + str(self.param['mod_prior']))
            else:
                print('Using ExoReL-R for giant gaseous planets')

        if MPIimport:
            MPI.COMM_WORLD.Barrier()  # wait for everybody to synchronize here

        parameters, n_params = retrieval_par_and_npar(self.param)
        if (self.param['gas_par_space'] == 'clr' or self.param['gas_par_space'] == 'centered_log_ratio') and self.param['mod_prior']:
            ppf = np.loadtxt(self.param['pkg_dir'] + 'forward_mod/Data/prior/prior_cube_' + str(len(self.param['fit_molecules'])) + 'gas.dat')

        self.param = MPI.COMM_WORLD.bcast(self.param, root=0)

        if MPIimport:
            MPI.COMM_WORLD.Barrier()  # wait for everybody to synchronize here

        def internal_model(cube, phi=None, n_obs=0, free_cld_calc=False, retrieval_mode=True):
            evaluation = {}
            par = 0
            if self.param['fit_p0'] and self.param['gas_par_space'] != 'partial_pressure':
                evaluation['P0'] = (10.0 ** cube[par])  # P0, surface pressure
                par += 1

            if self.param['fit_wtr_cld'] and not free_cld_calc:
                evaluation['pH2O'], evaluation['dH2O'], evaluation['crH2O'] = (10.0 ** cube[par]), (10.0 ** cube[par + 1]), (10.0 ** cube[par + 2])  # pH2O, dH2O, crH2O
                par += 3
            elif not self.param['fit_wtr_cld'] and free_cld_calc:
                par += 3

            if self.param['fit_amm_cld'] and not free_cld_calc:
                evaluation['pNH3'], evaluation['dNH3'], evaluation['crNH3'] = (10.0 ** cube[par]), (10.0 ** cube[par + 1]), (10.0 ** cube[par + 2])  # pNH3, dNH3, crNH3
                par += 3
            elif not self.param['fit_amm_cld'] and free_cld_calc:
                par += 3

            if self.param['gas_par_space'] == 'centered_log_ratio' or self.param['gas_par_space'] == 'clr':
                c_l_r = cube[par: par + len(self.param['fit_molecules'])]  # CLR Molecules
                c_l_r.append(-np.sum(np.array(cube[par: par + len(self.param['fit_molecules'])])))
                v_m_r = clr_inv(c_l_r)
                for m, mol in enumerate(self.param['fit_molecules']):
                    evaluation[mol] = v_m_r[m]
                    par += 1
                evaluation[self.param['gas_fill']] = 1.0
                for mol in self.param['fit_molecules']:
                    evaluation[self.param['gas_fill']] -= evaluation[mol]
            elif self.param['gas_par_space'] == 'volume_mixing_ratio' or self.param['gas_par_space'] == 'vmr':
                for mol in self.param['fit_molecules']:
                    evaluation[mol] = 10.0 ** cube[par]
                    par += 1
                evaluation[self.param['gas_fill']] = 1.0
                for mol in self.param['fit_molecules']:
                    evaluation[self.param['gas_fill']] -= evaluation[mol]
            elif self.param['gas_par_space'] == 'partial_pressure':
                evaluation['P0'] = np.sum(10.0 ** np.array(cube[par: par + len(self.param['fit_molecules'])]))
                for mol in self.param['fit_molecules']:
                    evaluation[mol] = (10.0 ** cube[par]) / evaluation['P0']
                    par += 1

            if self.param['fit_ag']:
                if self.param['surface_albedo_parameters'] == int(1):
                    evaluation['ag'] = cube[par] + 0.0  # Ag, surface albedo
                    par += 1
                elif self.param['surface_albedo_parameters'] == int(3):
                    for surf_alb in [1, 2]:
                        evaluation['ag' + str(surf_alb)] = cube[par + (surf_alb - 1)] + 0.0  # Ag, surface albedo
                    evaluation['ag_x1'] = cube[par + surf_alb] + 0.0
                    par += 3
                elif self.param['surface_albedo_parameters'] == int(5):
                    for surf_alb in [1, 2, 3]:
                        evaluation['ag' + str(surf_alb)] = cube[par + (surf_alb - 1)] + 0.0  # Ag, surface albedo
                    evaluation['ag_x1'] = cube[par + surf_alb] + 0.0
                    evaluation['ag_x2'] = cube[par + surf_alb + 1] + 0.0
                    par += 5

            if self.param['fit_T']:
                evaluation['Tp'] = cube[par] + 0.0  # Planetary temperature
                par += 1

            if self.param['fit_cld_frac']:
                self.param['cld_frac'] = (10.0 ** cube[par])  # cloud fraction
                par += 1

            if self.param['fit_g']:
                evaluation['gp'] = cube[par] + 0.0  # g
                par += 1
            if self.param['fit_Mp']:
                evaluation['Mp'] = cube[par] + 0.0  # Mp
                par += 1
            if self.param['fit_Rp']:
                evaluation['Rp'] = cube[par] + 0.0  # Rp
                par += 1

            if self.param['fit_p_size']:
                evaluation['p_size'] = (10. ** cube[par])  # p_size
                par += 1
            if self.param['fit_phi']:
                evaluation['phi'] = cube[par + n_obs] * math.pi / 180.  # phi

            if platform.system() == 'Darwin':
                return forward(self.param, evaluation=evaluation, phi=phi, n_obs=n_obs, retrieval_mode=retrieval_mode, core_number=MPIrank,
                               albedo_calc=self.param['albedo_calc'], fp_over_fs=self.param['fp_over_fs'], canc_metadata=True)
            else:
                try:
                    return forward(self.param, evaluation=evaluation, phi=phi, n_obs=n_obs, retrieval_mode=retrieval_mode, core_number=MPIrank,
                                   albedo_calc=self.param['albedo_calc'], fp_over_fs=self.param['fp_over_fs'], canc_metadata=True)
                except:
                    if MPIimport:
                        MPI.Finalize()
                        sys.exit()
                    else:
                        print('Some errors occurred in during the calculation of the forward model.')
                        sys.exit()

        def prior(cube, ndim, nparams):
            par = 0
            if self.param['fit_p0'] and self.param['gas_par_space'] != 'partial_pressure':
                cube[par] = (cube[par] * (self.param['p0_range'][1] - self.param['p0_range'][0])) + self.param['p0_range'][0]  # uniform prior between   3  :  11    -> P0, surface pressure
                par += 1
            if self.param['fit_wtr_cld']:
                cube[par] = (cube[par] * (self.param['ptopw_range'][1] - self.param['ptopw_range'][0])) + self.param['ptopw_range'][0]  # uniform prior between   0  :  8     -> P H2O cloud top [Pa]
                cube[par + 1] = (cube[par + 1] * (self.param['dcldw_range'][1] - self.param['dcldw_range'][0])) + self.param['dcldw_range'][0]  # uniform prior between   0  :  8.5   -> D H2O cloud [Pa]
                cube[par + 2] = (cube[par + 2] * (self.param['crh2o_range'][1] - self.param['crh2o_range'][0])) + self.param['crh2o_range'][0]  # uniform prior between -12  :  0     -> CR H2O
                par += 3

            if self.param['fit_amm_cld']:
                cube[par] = cube[par] * (self.param['ptopa_range'][1] - self.param['ptopa_range'][0]) + self.param['ptopa_range'][0]  # uniform prior between   0  :  8     -> P NH3 cloud top [Pa]
                cube[par + 1] = cube[par + 1] * (self.param['dclda_range'][1] - self.param['dclda_range'][0]) + self.param['dclda_range'][0]  # uniform prior between   0  :  8.5   -> D H2O cloud [Pa]
                cube[par + 2] = cube[par + 2] * (self.param['crnh3_range'][1] - self.param['crnh3_range'][0]) + self.param['crnh3_range'][0]  # uniform prior between -12  :  0     -> CR NH3
                par += 3

            for mol in self.param['fit_molecules']:
                if self.param['gas_par_space'] == 'centered_log_ratio' or self.param['gas_par_space'] == 'clr':
                    if self.param['mod_prior']:
                        cube[par] = ppf[find_nearest(ppf[:, 0], cube[par]), 1]  # modified prior for clr
                    else:
                        cube[par] = (cube[par] * (self.param['clr' + mol + '_range'][1] - self.param['clr' + mol + '_range'][0])) + self.param['clr' + mol + '_range'][0]  # uniform clr prior between -25 : 25
                elif self.param['gas_par_space'] == 'volume_mixing_ratio' or self.param['gas_par_space'] == 'vmr':
                    cube[par] = (cube[par] * (self.param['vmr' + mol + '_range'][1] - self.param['vmr' + mol + '_range'][0])) + self.param['vmr' + mol + '_range'][0]  # uniform vmr prior between -12 : 0
                elif self.param['gas_par_space'] == 'partial_pressure':
                    cube[par] = (cube[par] * (self.param['pp' + mol + '_range'][1] - self.param['pp' + mol + '_range'][0])) + self.param['pp' + mol + '_range'][0]  # uniform partial pressure prior between -10 : 10
                par += 1

            if self.param['fit_ag']:
                if self.param['surface_albedo_parameters'] == int(1):
                    cube[par] = (cube[par] * (self.param['ag_range'][1] - self.param['ag_range'][0])) + self.param['ag_range'][0]  # uniform prior between   0.0  :  0.5     -> Ag, surface albedo
                    par += 1
                elif self.param['surface_albedo_parameters'] == int(3):
                    for surf_alb in [1, 2]:
                        cube[par + (surf_alb - 1)] = (cube[par + (surf_alb - 1)] * (self.param['ag' + str(surf_alb) + '_range'][1] - self.param['ag' + str(surf_alb) + '_range'][0])) + self.param['ag' + str(surf_alb) + '_range'][0]
                    cube[par + surf_alb] = (cube[par + surf_alb] * (self.param['ag_x1_range'][1] - self.param['ag_x1_range'][0])) + self.param['ag_x1_range'][0]
                    par += 3
                elif self.param['surface_albedo_parameters'] == int(5):
                    for surf_alb in [1, 2, 3]:
                        cube[par + (surf_alb - 1)] = (cube[par + (surf_alb - 1)] * (self.param['ag' + str(surf_alb) + '_range'][1] - self.param['ag' + str(surf_alb) + '_range'][0])) + self.param['ag' + str(surf_alb) + '_range'][0]
                    cube[par + surf_alb] = (cube[par + surf_alb] * (self.param['ag_x1_range'][1] - self.param['ag_x1_range'][0])) + self.param['ag_x1_range'][0]
                    cube[par + surf_alb + 1] = cube[par + surf_alb] + (cube[par + surf_alb + 1] * (self.param['ag_x2_range'][1] - self.param['ag_x2_range'][0])) + self.param['ag_x2_range'][0]
                    par += 5
                if self.param['fit_T']:
                    cube[par] = (cube[par] * (self.param['tp_range'][1] - self.param['tp_range'][0])) + self.param['tp_range'][0]  # uniform prior between   0  :  700   -> Tp, planetary temperature
                    par += 1

            if self.param['fit_cld_frac']:
                cube[par] = (cube[par] * (self.param['cld_frac_range'][1] - self.param['cld_frac_range'][0])) + self.param['cld_frac_range'][0]  # uniform prior between   -3.0  :  0.0     -> Log(clf_frac), cloud fraction
                par += 1

            if self.param['fit_g']:
                cube[par] = (cube[par] * (self.param['gp_range'][1] - self.param['gp_range'][0])) + self.param['gp_range'][0]  # uniform prior between   0.5  :  6   -> g [m/s2]
                par += 1

            if self.param['fit_Mp'] and self.param['fit_Rp']:
                if self.param['Rp_prior_type'] != 'R_M_prior' and self.param['Mp_prior_type'] != 'M_R_prior':
                    cube[par] = Mp_prior(self.param, cube[par])  # Mass prior - independent
                    cube[par + 1] = Rp_prior(self.param, cube[par + 1])  # Radius prior - independent
                    par += 2
                elif self.param['Rp_prior_type'] != 'R_M_prior' and self.param['Mp_prior_type'] == 'M_R_prior':
                    cube[par] = Mp_prior(self.param, cube[par])  # Mass prior - independent
                    cube[par + 1] = Rp_prior(self.param, cube[par + 1], mp_value=cube[par])  # Radius prior - 2D prior
                    par += 2
                elif self.param['Rp_prior_type'] == 'R_M_prior' and self.param['Mp_prior_type'] != 'M_R_prior':
                    cube[par + 1] = Rp_prior(self.param, cube[par + 1])  # Radius prior - independent
                    cube[par] = Mp_prior(self.param, cube[par], rp_value=cube[par + 1])  # Mass prior - 2D prior
                    par += 2
            elif self.param['fit_Mp'] and not self.param['fit_Rp']:
                cube[par] = Mp_prior(self.param, cube[par])  # Mass prior
                par += 1
            elif self.param['fit_Rp'] and not self.param['fit_Mp']:
                cube[par] = Rp_prior(self.param, cube[par])  # Radius prior
                par += 1

            if self.param['fit_p_size']:
                cube[par] = (cube[par] * (self.param['p_size_range'][1] - self.param['p_size_range'][0])) + self.param['p_size_range'][0]  # Particle size uniform prior
                par += 1

            if self.param['fit_phi']:
                if self.param['obs_numb'] is None:
                    cube[par] = (cube[par] * (self.param['phi_range'][1] - self.param['phi_range'][0])) + self.param['phi_range'][0]  # uniform prior between   0  :  180   -> deg
                    par += 1
                else:
                    for obse in range(0, self.param['obs_numb']):
                        cube[par] = (cube[par] * (self.param['phi_range'][1] - self.param['phi_range'][0])) + self.param['phi_range'][0]  # uniform prior between   0  :  180   -> deg
                        par += 1

        def loglike(cube, ndim, nparams):
            if self.param['obs_numb'] is None:
                model = internal_model(cube)

                if self.param['fit_wtr_cld'] and self.param['cld_frac'] != 1.0:
                    self.param['fit_wtr_cld'] = False
                    model_no_cld = internal_model(cube, free_cld_calc=True)
                    self.param['fit_wtr_cld'] = True
                    model = (self.param['cld_frac'] * model) + ((1.0 - self.param['cld_frac']) * model_no_cld)

                chi = (self.param['spectrum']['Fplanet'] - model) / self.param['spectrum']['error_p']
                loglikelihood = (-1.) * np.sum(np.log(self.param['spectrum']['error_p'] * np.sqrt(2.0 * math.pi))) - 0.5 * np.sum(chi * chi)
            else:
                loglikelihood = 0.0
                for obs in range(0, self.param['obs_numb']):
                    if self.param['fit_phi']:
                        chi = (self.param['spectrum'][str(obs)]['Fplanet'] - internal_model(cube, phi=None, n_obs=obs)) / self.param['spectrum'][str(obs)]['error_p']
                    else:
                        chi = (self.param['spectrum'][str(obs)]['Fplanet'] - internal_model(cube, phi=self.param['phi' + str(obs)], n_obs=obs)) / self.param['spectrum'][str(obs)]['error_p']
                    loglikelihood += (-1.) * np.sum(np.log(self.param['spectrum'][str(obs)]['error_p'] * np.sqrt(2.0 * math.pi))) - 0.5 * np.sum(chi * chi)

            return loglikelihood

        if MPIimport and MPIrank == 0:
            time1 = time.time()

        pymultinest.run(LogLikelihood=loglike,
                        Prior=prior,
                        n_dims=n_params,
                        multimodal=self.param['multimodal'],
                        max_modes=self.param['max_modes'],
                        outputfiles_basename=self.param['out_dir'] + self.param['name_p'] + '_',
                        importance_nested_sampling=False,
                        evidence_tolerance=self.param['ev_tolerance'],
                        n_live_points=self.param['nlive_p'],
                        resume=self.param['multinest_resume'],
                        verbose=self.param['multinest_verbose'],
                        init_MPI=False)

        if MPIimport and MPIrank == 0:  # Plot Nest_spectrum
            time2 = time.time()
            elapsed((time2 - time1) * (10 ** 9))

        prefix = self.param['out_dir'] + self.param['name_p'] + '_'
        if MPIimport and MPIrank == 0:
            json.dump(parameters, open(prefix + 'params.json', 'w'))  # save parameter names

        ### PRODUCE PLOTS FROM HERE --- POST-PROCESSING ###
        self.param['model_n_par'] = len(parameters)
        multinest_results = pymultinest.Analyzer(n_params=self.param['model_n_par'], outputfiles_basename=prefix, verbose=False)
        mc_samp = multinest_results.get_equal_weighted_posterior()[:, :-1]

        if self.param['calc_likelihood_data']:
            self.calc_spectra(mc_samp)

            if MPIimport:
                MPI.COMM_WORLD.Barrier()  # wait for everybody to synchronize here

            if MPIimport and MPIrank == 0:
                if platform.system() != 'Darwin':
                    time.sleep(600)
                rank_0 = np.loadtxt(self.param['out_dir'] + 'loglikelihood_per_datapoint/loglike_0.dat')
                rank_0_spec = np.loadtxt(self.param['out_dir'] + 'loglikelihood_per_datapoint/samples_0.dat')
                for i in range(1, MPIsize):
                    rank_n = np.loadtxt(self.param['out_dir'] + 'loglikelihood_per_datapoint/loglike_' + str(i) + '.dat')
                    rank_n_spec = np.loadtxt(self.param['out_dir'] + 'loglikelihood_per_datapoint/samples_' + str(i) + '.dat')
                    rank_0 = np.concatenate((rank_0, rank_n), axis=0)
                    rank_0_spec = np.concatenate((rank_0_spec, rank_n_spec[:, 1:]), axis=1)
                np.savetxt(self.param['out_dir'] + 'loglike_per_datapoint.dat', rank_0)
                np.savetxt(self.param['out_dir'] + 'random_samples.dat', rank_0_spec)
                os.system('rm -rf ' + self.param['out_dir'] + 'loglikelihood_per_datapoint/')

                self.param['spec_sample'] = rank_0_spec + 0.0
                del rank_0_spec, rank_0

        if MPIimport and MPIrank == 0:  # Plot Nest_spectrum
            if self.param['filter_multi_solutions']:
                s, mds = self.filter_pymultinest_modes(multinest_results)
                mds_orig = len(multinest_results.get_stats()['modes'])
            else:
                s = multinest_results.get_stats()
                mds_orig = mds = len(s['modes'])

            if self.param['plot_models']:
                if mds < 2:
                    # s = multinest_results.get_best_fit()
                    # cube = s['parameters']
                    # cube = np.array([cube, np.ones(len(s['parameters']))]).T

                    # cube = []
                    # for p, m in zip(parameters, s['marginals']):
                    #     cube.append(m['median'])
                    # cube = np.array([cube, np.ones(len(parameters))]).T

                    cube = np.ones((len(s['modes'][0]['maximum a posterior']), mds))
                    cube[:, 0] = list(s['modes'][0]['maximum a posterior'])

                    self.plot_nest_spec(cube[:, 0])
                    plot_chemistry(self.param, solutions=None)
                    if self.param['surface_albedo_parameters'] > 1:
                        self.plot_surface_albedo()
                    if self.param['plot_contribution'] and self.param['obs_numb'] is None:
                        self.plot_contribution(cube[:, 0])
                else:
                    cube = np.ones((len(s['modes'][0]['maximum a posterior']), mds))
                    for i in range(0, mds):
                        cube[:, i] = list(s['modes'][i]['maximum a posterior'])

                        self.plot_nest_spec(cube[:, i], solutions=i + 1)
                        plot_chemistry(self.param, solutions=i + 1)
                        if self.param['surface_albedo_parameters'] > 1:
                            self.plot_surface_albedo(solutions=i + 1)
                        if self.param['plot_contribution'] and self.param['obs_numb'] is None:
                            self.plot_contribution(cube[:, i], solutions=i + 1)

                if self.param['spectrum']['bins']:
                    data_spec = np.array([self.param['spectrum']['wl_low'], self.param['spectrum']['wl_high'], self.param['spectrum']['wl'], self.param['spectrum']['Fplanet'], self.param['spectrum']['error_p']]).T
                else:
                    data_spec = np.array([self.param['spectrum']['wl'], self.param['spectrum']['Fplanet'], self.param['spectrum']['error_p']]).T
                np.savetxt(self.param['out_dir'] + 'data_spectrum.dat', data_spec)

            if self.param['plot_posterior']:
                self.plot_posteriors(prefix, multinest_results, parameters, mds_orig)

        if MPIimport:
            MPI.Finalize()

    def filter_pymultinest_modes(self, mres):
        mds = len(mres.get_stats()['modes'])
        if mds == 1:
            return mres.get_stats(), mds
        else:
            max_ev, max_idx = 0, 0
            for i in range(0, mds):
                if mres.get_stats()['modes'][i]['local log-evidence'] > max_ev:
                    max_ev = mres.get_stats()['modes'][i]['local log-evidence'] + 0.0
                    max_idx = i
                else:
                    pass

            filtered_modes = [mres.get_stats()['modes'][max_idx]]
            filtered_modes[len(filtered_modes) - 1]['index'] = len(filtered_modes) - 1
            for i in range(0, mds):
                if i == max_idx:
                    pass
                elif (max_ev - mres.get_stats()['modes'][i]['local log-evidence']) < 11.0:
                    filtered_modes.append(mres.get_stats()['modes'][i])
                    filtered_modes[len(filtered_modes) - 1]['index'] = len(filtered_modes) - 1
                else:
                    pass

            s = mres.get_stats()
            s['modes'] = filtered_modes

            if mds - len(s['modes']) > 0:
                print("\n" + str(mds - len(s['modes'])) + " modes have been filtered due to low significance")
            return s, len(s['modes'])

    def cube_to_param(self, cube, n_obs=0, free_cld_calc=False):
        par = 0
        if self.param['fit_p0'] and self.param['gas_par_space'] != 'partial_pressure':
            self.param['P0'] = 10. ** cube[par]
            par += 1

        if self.param['fit_wtr_cld'] and not free_cld_calc:
            self.param['Pw_top'] = 10. ** cube[par]
            self.param['cldw_depth'] = 10. ** cube[par + 1]
            self.param['CR_H2O'] = 10. ** cube[par + 2]
            par += 3
        elif not self.param['fit_wtr_cld'] and free_cld_calc:
            par += 3

        if self.param['fit_amm_cld'] and not free_cld_calc:
            self.param['Pa_top'] = 10. ** cube[par]
            self.param['clda_depth'] = 10. ** cube[par + 1]
            self.param['CR_NH3'] = 10. ** cube[par + 2]
            par += 3
        elif not self.param['fit_amm_cld'] and free_cld_calc:
            par += 3

        if self.param['gas_par_space'] == 'centered_log_ratio' or self.param['gas_par_space'] == 'clr':
            clr = {}
            for mol in self.param['fit_molecules']:
                clr[mol] = cube[par]
                par += 1
            self.param, _ = clr_to_vmr(self.param, clr)
        elif self.param['gas_par_space'] == 'volume_mixing_ratio' or self.param['gas_par_space'] == 'vmr':
            for mol in self.param['fit_molecules']:
                self.param['vmr_' + mol] = 10. ** cube[par]
                par += 1
            self.param['vmr_' + self.param['gas_fill']] = 1.0
            for mol in self.param['fit_molecules']:
                self.param['vmr_' + self.param['gas_fill']] -= self.param['vmr_' + mol]
        elif self.param['gas_par_space'] == 'partial_pressure':
            self.param['P0'] = np.sum(10.0 ** np.array(cube[par: par + len(self.param['fit_molecules'])]))
            for mol in self.param['fit_molecules']:
                self.param['vmr_' + mol] = (10.0 ** cube[par]) / self.param['P0']
                par += 1

        self.param['P'] = 10. ** np.arange(0.0, np.log10(self.param['P0']) + 0.01, step=0.01)
        if self.param['fit_amm_cld']:
            self.param['vmr_NH3'] = cloud_pos(self.param, condensed_gas='NH3')
            self.param = adjust_VMR(self.param, all_gases=self.param['adjust_VMR_gases'], condensed_gas='NH3')
        self.param['vmr_H2O'] = cloud_pos(self.param, condensed_gas='H2O')
        self.param = adjust_VMR(self.param, all_gases=self.param['adjust_VMR_gases'], condensed_gas='H2O')
        if self.param['O3_earth']:
            self.param['vmr_O3'] = ozone_earth_mask(self.param)
        self.param = calc_mean_mol_mass(self.param)

        if self.param['fit_ag']:
            if self.param['surface_albedo_parameters'] == int(1):
                self.param['Ag'] = cube[par] + 0.0  # Ag, surface albedo
                par += 1
            elif self.param['surface_albedo_parameters'] == int(3):
                for surf_alb in [1, 2]:
                    self.param['Ag' + str(surf_alb)] = cube[par + (surf_alb - 1)] + 0.0  # Ag, surface albedo
                self.param['Ag_x1'] = cube[par + surf_alb] + 0.0
                par += 3
            elif self.param['surface_albedo_parameters'] == int(5):
                for surf_alb in [1, 2, 3]:
                    self.param['Ag' + str(surf_alb)] = cube[par + (surf_alb - 1)] + 0.0  # Ag, surface albedo
                self.param['Ag_x1'] = cube[par + surf_alb] + 0.0
                self.param['Ag_x2'] = cube[par + surf_alb + 1] + 0.0
                par += 5

        if self.param['fit_T']:
            self.param['Tp'] = cube[par] + 0.0  # Planetary temperature
            par += 1

        if self.param['fit_cld_frac']:
            self.param['cld_frac'] = (10.0 ** cube[par])  # Cloud fraction
            par += 1

        if self.param['fit_g'] and self.param['fit_Mp'] and not self.param['fit_Rp']:
            self.param['gp'] = 10. ** (cube[par] - 2.0)  # g is in m/s2 but it was defined in cgs
            self.param['Mp'] = cube[par + 1]  # Mp is in M_jup
            self.param['Rp'] = (np.sqrt((const.G.value * const.M_jup.value * self.param['Mp']) / self.param['gp'])) / const.R_jup.value  # Rp is in R_jup
            par += 2
        elif self.param['fit_g'] and self.param['fit_Rp'] and not self.param['fit_Mp']:
            self.param['gp'] = (10. ** (cube[par] - 2.0))  # g is in m/s2 but it was defined in cgs
            self.param['Rp'] = cube[par + 1]  # Rp is in R_jup
            self.param['Mp'] = ((self.param['gp'] * ((self.param['Rp'] * const.R_jup.value) ** 2.)) / const.G.value) / const.M_jup.value  # Mp is in M_jup
            par += 2
        elif self.param['fit_Mp'] and self.param['fit_Rp'] and not self.param['fit_g']:
            self.param['Mp'] = cube[par]  # Mp is in M_jup
            self.param['Rp'] = cube[par + 1]  # Rp is in R_jup
            self.param['gp'] = (const.G.value * const.M_jup.value * self.param['Mp']) / ((const.R_jup.value * self.param['Rp']) ** 2.)  # g is in m/s2
            par += 2
        elif self.param['fit_g'] and not self.param['fit_Mp'] and not self.param['fit_Rp'] and self.param['Mp'] is not None:
            self.param['gp'] = (10. ** (cube[par] - 2.0))  # g is in m/s2 but it was defined in cgs
            self.param['Rp'] = (np.sqrt((const.G.value * const.M_jup.value * self.param['Mp']) / self.param['gp'])) / const.R_jup.value  # Rp is in R_jup
            par += 1
        elif self.param['fit_Rp'] and not self.param['fit_Mp'] and not self.param['fit_g'] and self.param['Mp'] is not None:
            self.param['Rp'] = cube[par]  # Rp is in R_jup
            self.param['gp'] = (const.G.value * const.M_jup.value * self.param['Mp']) / ((const.R_jup.value * self.param['Rp']) ** 2.)  # g is in m/s2
            par += 1
        elif not self.param['fit_g'] and not self.param['fit_Mp'] and not self.param['fit_Rp']:
            if not self.param['Mp_provided']:
                self.param['Mp'] = ((self.param['gp'] * ((self.param['Rp'] * const.R_jup.value) ** 2.)) / const.G.value) / const.M_jup.value  # Mp is in M_jup
            if not self.param['Rp_provided']:
                self.param['Rp'] = (np.sqrt((const.G.value * const.M_jup.value * self.param['Mp']) / self.param['gp'])) / const.R_jup.value  # Rp is in R_jup
            if not self.param['gp_provided']:
                self.param['gp'] = (const.G.value * const.M_jup.value * self.param['Mp']) / ((const.R_jup.value * self.param['Rp']) ** 2.)  # g is in m/s2

        if self.param['fit_p_size']:  # particle size, constant
            self.param['p_size'] = (10. ** cube[par])
            par += 1

        if self.param['fit_phi']:
            self.param['phi'] = cube[par + n_obs] * math.pi / 180.  # phi

        self.param['core_number'] = None

    def plot_nest_spec(self, cube, solutions=None):
        self.cube_to_param(cube)
        if self.param['obs_numb'] is None:
            fig = plt.figure(figsize=(8, 5))

            plt.errorbar(self.param['spectrum']['wl'], self.param['spectrum']['Fplanet'], yerr=self.param['spectrum']['error_p'],
                         linestyle='', linewidth=0.5, color='black', marker='o', markerfacecolor='red', markersize=4, capsize=1.75, label='Data')

            mod = FORWARD_MODEL(self.param, retrieval=False, canc_metadata=True)
            alb_wl, alb = mod.run_forward()
            alb_wl *= 10. ** (-3.)

            if self.param['fit_wtr_cld'] and self.param['cld_frac'] != 1.0:
                alb = self.adjust_for_cld_frac(alb, cube)
                self.cube_to_param(cube)

            _, model = model_finalizzation(self.param, alb_wl, alb, planet_albedo=self.param['albedo_calc'], fp_over_fs=self.param['fp_over_fs'])
            plt.plot(self.param['spectrum']['wl'], model, linestyle='', color='black', marker='d', markerfacecolor='blue', markersize=4)

            if self.param['mol_custom_wl']:
                new_wl = np.loadtxt(self.param['pkg_dir'] + 'forward_mod/Data/wl_bins/bins_02_50_R500.dat')
                new_wl_central = np.mean(new_wl, axis=1)
                start = 0
                stop = len(new_wl_central) - 1
            else:
                new_wl = np.loadtxt(self.param['pkg_dir'] + 'forward_mod/Data/wl_bins/bins_02_20_R500.dat')
                new_wl_central = np.mean(new_wl, axis=1)
                start = find_nearest(new_wl_central, min(self.param['spectrum']['wl']) - 0.05)
                stop = find_nearest(new_wl_central, max(self.param['spectrum']['wl']) + 0.05)

            if self.param['spectrum']['bins']:
                temp = np.array([self.param['spectrum']['wl_low'], self.param['spectrum']['wl_high'], self.param['spectrum']['wl']]).T
            else:
                temp = self.param['spectrum']['wl']
            self.param['spectrum']['wl'] = new_wl_central[start:stop]
            self.param['spectrum']['wl_low'] = new_wl[start:stop, 0]
            self.param['spectrum']['wl_high'] = new_wl[start:stop, 1]

            temp_min, temp_max = self.param['min_wl'] + 0.0, self.param['max_wl'] + 0.0
            self.param['min_wl'] = min(self.param['spectrum']['wl'])
            self.param['max_wl'] = max(self.param['spectrum']['wl'])
            self.param['start_c_wl_grid'] = find_nearest(self.param['wl_C_grid'], self.param['min_wl']) - 35
            self.param['stop_c_wl_grid'] = find_nearest(self.param['wl_C_grid'], self.param['max_wl']) + 35

            mod = FORWARD_MODEL(self.param, retrieval=False, canc_metadata=True)
            alb_wl, alb = mod.run_forward()
            alb_wl *= 10. ** (-3.)

            if self.param['fit_wtr_cld'] and self.param['cld_frac'] != 1.0:
                alb = self.adjust_for_cld_frac(alb, cube)
                self.cube_to_param(cube)

            wl, model = model_finalizzation(self.param, alb_wl, alb, planet_albedo=self.param['albedo_calc'], fp_over_fs=self.param['fp_over_fs'])
            plt.plot(wl, model, color='#404784', linewidth=0.5, label='MAP solution, R=500')

            best_fit = np.array([wl, model]).T

            if os.path.isfile(self.param['out_dir'] + 'random_samples.dat'):
                fl = np.loadtxt(self.param['out_dir'] + 'random_samples.dat')

                best_fit = np.concatenate((best_fit, np.array([best_fit[:, 1] + (np.quantile(fl[:, 1:], [0.16, 0.84], axis=1)[1] - np.quantile(fl[:, 1:], 0.5, axis=1))]).T), axis=1)
                best_fit = np.concatenate((best_fit, np.array([best_fit[:, 1] + (np.quantile(fl[:, 1:], [0.16, 0.84], axis=1)[0] - np.quantile(fl[:, 1:], 0.5, axis=1))]).T), axis=1)
                best_fit = np.concatenate((best_fit, np.array([best_fit[:, 1] + (np.quantile(fl[:, 1:], [0.0225, 0.9775], axis=1)[1] - np.quantile(fl[:, 1:], 0.5, axis=1))]).T), axis=1)
                best_fit = np.concatenate((best_fit, np.array([best_fit[:, 1] + (np.quantile(fl[:, 1:], [0.0225, 0.9775], axis=1)[0] - np.quantile(fl[:, 1:], 0.5, axis=1))]).T), axis=1)
                best_fit = np.concatenate((best_fit, np.array([best_fit[:, 1] + (np.quantile(fl[:, 1:], [0.00135, 0.99865], axis=1)[1] - np.quantile(fl[:, 1:], 0.5, axis=1))]).T), axis=1)
                best_fit = np.concatenate((best_fit, np.array([best_fit[:, 1] + (np.quantile(fl[:, 1:], [0.00135, 0.99865], axis=1)[0] - np.quantile(fl[:, 1:], 0.5, axis=1))]).T), axis=1)

                for i in range(0, len(best_fit[:, 0])):
                    for j in range(2, 8):
                        if best_fit[i, j] < 0.0:
                            best_fit[i, j] = 0.0

                plt.fill_between(fl[:, 0], best_fit[:, 2], best_fit[:, 3], ec=('#404784', 0.0), fc=('#404784', 0.25), label='1$\sigma$')
                plt.fill_between(fl[:, 0], best_fit[:, 4], best_fit[:, 5], ec=('#404784', 0.0), fc=('#404784', 0.5), label='2$\sigma$')
                plt.fill_between(fl[:, 0], best_fit[:, 6], best_fit[:, 7], ec=('#404784', 0.0), fc=('#404784', 0.75), label='3$\sigma$')

                del fl

            if solutions is None:
                np.savetxt(self.param['out_dir'] + 'Best_fit.dat', best_fit)
            else:
                np.savetxt(self.param['out_dir'] + 'Best_fit_' + str(solutions) + '.dat', best_fit)

            if self.param['spectrum']['bins']:
                self.param['spectrum']['wl'] = temp[:, 2]
                self.param['spectrum']['wl_low'] = temp[:, 0]
                self.param['spectrum']['wl_high'] = temp[:, 1]
            else:
                self.param['spectrum']['wl'] = temp + 0.0

            self.param['min_wl'] = copy.deepcopy(temp_min)
            self.param['max_wl'] = copy.deepcopy(temp_max)
            self.param['start_c_wl_grid'] = find_nearest(self.param['wl_C_grid'], self.param['min_wl']) - 35
            self.param['stop_c_wl_grid'] = find_nearest(self.param['wl_C_grid'], self.param['max_wl']) + 35

            plt.legend()
            plt.xlabel('Wavelength [$\mu$m]')
            if self.param['albedo_calc']:
                plt.ylabel('Albedo')
            elif self.param['fp_over_fs']:
                plt.ylabel('Contrast Ratio (F$_p$/F$_{\star}$)')
            else:
                plt.ylabel('Planetary flux [W/m$^2$]')
            fig.tight_layout()

        else:

            fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
            new_wl = np.loadtxt(self.param['pkg_dir'] + 'forward_mod/Data/wl_bins/bins_02_20_R500.dat')
            new_wl_central = np.mean(new_wl, axis=1)
            self.param['spectrum']['bins'] = False

            mod = FORWARD_MODEL(self.param, retrieval=False, canc_metadata=True)

            for obs in range(0, self.param['obs_numb']):
                self.cube_to_param(cube, n_obs=obs)
                alb_wl, alb = mod.run_forward()
                alb_wl *= 10. ** (-3.)

                if self.param['fit_cld_frac'] and self.param['fit_wtr_cld'] and self.param['cld_frac'] != 1.0:
                    alb = self.adjust_for_cld_frac(alb, cube)
                    self.cube_to_param(cube, n_obs=obs)

                _, model = model_finalizzation(self.param, alb_wl, alb, planet_albedo=self.param['albedo_calc'], fp_over_fs=self.param['fp_over_fs'], n_obs=obs)

                axs[obs].plot(self.param['spectrum'][str(obs)]['wl'], model, linestyle='', color='black', marker='d', markerfacecolor='blue', markersize=4)

                if self.param['fit_phi']:
                    lab = 'Input spectrum ' + str(obs)
                else:
                    lab = 'Data, $\phi=$' + str(self.param['phi' + str(obs)] * 180. / math.pi) + ' deg'

                axs[obs].errorbar(self.param['spectrum'][str(obs)]['wl'], self.param['spectrum'][str(obs)]['Fplanet'], yerr=self.param['spectrum'][str(obs)]['error_p'],
                                  linestyle='', linewidth=0.5, color='black', marker='o', markerfacecolor='red', markersize=4, capsize=1.75,
                                  label=lab)

                start = find_nearest(new_wl_central, min(self.param['spectrum'][str(obs)]['wl']) - 0.05)
                stop = find_nearest(new_wl_central, max(self.param['spectrum'][str(obs)]['wl']) + 0.05)

                temp = self.param['spectrum'][str(obs)]['wl'] + 0.0
                self.param['spectrum'][str(obs)]['wl'] = new_wl_central[start:stop]

                alb_wl, alb = mod.run_forward()
                alb_wl *= 10. ** (-3.)

                if self.param['fit_cld_frac'] and self.param['fit_wtr_cld'] and self.param['cld_frac'] != 1.0:
                    alb = self.adjust_for_cld_frac(alb, cube)
                    self.cube_to_param(cube, n_obs=obs)

                wl, model = model_finalizzation(self.param, alb_wl, alb, planet_albedo=self.param['albedo_calc'], fp_over_fs=self.param['fp_over_fs'], n_obs=obs)

                axs[obs].plot(wl, model, label='MAP solution, R=500')

                self.param['spectrum'][str(obs)]['wl'] = temp + 0.0

                axs[obs].legend()
                axs[obs].set_ylim([-0.2 * min(model), max(model) + 0.2 * max(model)])

            if self.param['albedo_calc']:
                fig.text(0.04, 0.5, 'Albedo', va='center', rotation='vertical')
            else:
                if self.param['fp_over_fs']:
                    fig.text(0.04, 0.5, 'F$_p$/F$_{\star}$', va='center', rotation='vertical')
                else:
                    fig.text(0.04, 0.5, 'Planetary flux [W/m$^2$]', va='center', rotation='vertical')

            # axs[obs].set_xlim([0.45, 1.0])
            fig.text(0.5, 0.04, 'Wavelength [$\mu$m]', ha='center')

        if solutions is None:
            plt.savefig(self.param['out_dir'] + 'Nest_spectrum.pdf')
        else:
            plt.savefig(self.param['out_dir'] + 'Nest_spectrum (solution ' + str(solutions) + ').pdf')
        plt.close()

    def plot_surface_albedo(self, solutions=None):
        # Define parameters for the step function
        x1 = self.param['Ag_x1'] + 0.0
        a1, a2 = self.param['Ag1'] + 0.0, self.param['Ag2'] + 0.0
        if self.param['surface_albedo_parameters'] == int(5):
            x2 = self.param['Ag_x2'] + 0.0  # wavelength cutoffs in microns
            a3 = self.param['Ag3'] + 0.0  # albedo values for each region

        # Create data for plotting
        wavelength = np.linspace(self.param['min_wl'], self.param['max_wl'], 1000)  # x-axis: wavelength in microns

        # Create the step function
        step_albedo = np.zeros_like(wavelength)
        step_albedo[wavelength < x1] = a1
        if self.param['surface_albedo_parameters'] == int(3):
            step_albedo[wavelength >= x1] = a2
        elif self.param['surface_albedo_parameters'] == int(5):
            step_albedo[(wavelength >= x1) & (wavelength < x2)] = a2
            step_albedo[wavelength >= x2] = a3

        # Create a high-quality figure
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

        # Plot with enhanced styling
        ax.plot(wavelength, step_albedo, linewidth=3, color='#ff7f0e', label='Retrieved surface albedo function')

        # Fill areas under curves for visual enhancement
        ax.fill_between(wavelength, 0, step_albedo, alpha=0.3, color='#ff7f0e')

        # Add vertical lines at transition points
        ax.axvline(x=x1, color='#9467bd', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Transition $\lambda_1$: {np.round(x1,2)} $\mu$m')
        if self.param['surface_albedo_parameters'] == int(5):
            ax.axvline(x=x2, color='#d62728', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Transition $\lambda_2$: {np.round(x2,2)} $\mu$m')

        # Annotations for step values
        ax.annotate(f'a$_1$ = {np.round(a1,2)}', xy=((self.param['min_wl'] + x1) / 2, a1), xytext=((self.param['min_wl'] + x1) / 2, a1 + 0.04),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                    fontsize=12)
        if self.param['surface_albedo_parameters'] == int(3):
            ax.annotate(f'a$_2$ = {np.round(a2, 2)}', xy=((x1 + self.param['max_wl']) / 2, a2), xytext=((x1 + self.param['max_wl']) / 2, a2 + 0.04),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                        fontsize=12)
        elif self.param['surface_albedo_parameters'] == int(5):
            ax.annotate(f'a$_2$ = {np.round(a2,2)}', xy=((x1 + x2) / 2, a2), xytext=((x1 + x2) / 2, a2 + 0.04),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                        fontsize=12)
            ax.annotate(f'a$_3$ = {np.round(a3,2)}', xy=((x2 + self.param['max_wl']) / 2, a3), xytext=((x2 + self.param['max_wl']) / 2, a3 + 0.04),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                        fontsize=12)

        # Enhance the axes and labels
        ax.set_xlabel('Wavelength [$\mu$m]', fontsize=14, fontweight='bold')
        ax.set_ylabel('Surface Albedo', fontsize=14, fontweight='bold')

        # Set axis limits with a bit of padding
        # ax.set_xlim(0.25, 1.85)
        if self.param['surface_albedo_parameters'] == int(3):
            ax.set_ylim(-0.02, np.max([a1, a2]) + 0.1)
        elif self.param['surface_albedo_parameters'] == int(5):
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
            plt.savefig(self.param['out_dir'] + 'Nest_surface_albedo.pdf')
        else:
            plt.savefig(self.param['out_dir'] + 'Nest_surface_albedo (solution ' + str(solutions) + ').pdf')

        plt.close()

    def plot_contribution(self, cube, solutions=None):
        if solutions is None:
            if not os.path.isdir(self.param['out_dir'] + 'contr_comp/'):
                os.mkdir(self.param['out_dir'] + 'contr_comp/')
            contr_out = 'contr_comp/'
        else:
            if not os.path.isdir(self.param['out_dir'] + 'contr_comp_' + str(solutions) + '/'):
                os.mkdir(self.param['out_dir'] + 'contr_comp_' + str(solutions) + '/')
            contr_out = 'contr_comp_' + str(solutions) + '/'
        fig = plt.figure(figsize=(12, 5))

        new_wl = np.loadtxt(self.param['pkg_dir'] + 'forward_mod/Data/wl_bins/bins_02_20_R500.dat')
        new_wl_central = np.mean(new_wl, axis=1)
        is_bins = self.param['spectrum']['bins']
        self.param['spectrum']['bins'] = False

        if self.param['mol_custom_wl']:
            start = 0
            stop = len(new_wl_central) - 1
        else:
            start = find_nearest(new_wl_central, min(self.param['spectrum']['wl']) - 0.05)
            stop = find_nearest(new_wl_central, max(self.param['spectrum']['wl']) + 0.05)

        temp_wl = self.param['spectrum']['wl'] + 0.0
        self.param['spectrum']['wl'] = new_wl_central[start:stop]
        self.param['start_c_wl_grid'] = find_nearest(self.param['wl_C_grid'], min(self.param['spectrum']['wl'])) - 35
        self.param['stop_c_wl_grid'] = find_nearest(self.param['wl_C_grid'], max(self.param['spectrum']['wl'])) + 35

        self.param['contribution'] = True
        for mol in self.param['fit_molecules']:
            print('Plotting the contribution of ' + str(mol) + ' : VMR -> ' + str(self.param['vmr_' + mol][-1]))
            self.param['mol_contr'] = mol
            mod = FORWARD_MODEL(self.param, retrieval=False, canc_metadata=True)
            alb_wl, alb = mod.run_forward()
            alb_wl *= 10. ** (-3.)

            if self.param['fit_wtr_cld'] and self.param['cld_frac'] != 1.0:
                alb = self.adjust_for_cld_frac(alb, cube)
                self.cube_to_param(cube)

            wl, model = model_finalizzation(self.param, alb_wl, alb, planet_albedo=self.param['albedo_calc'], fp_over_fs=self.param['fp_over_fs'])
            plt.plot(wl, model, label=mol)
            single_contr = np.array([wl, model]).T
            np.savetxt(self.param['out_dir'] + contr_out + mol + '.dat', single_contr)
        self.param['mol_contr'] = None

        mod = FORWARD_MODEL(self.param, retrieval=False, canc_metadata=True)
        alb_wl, alb = mod.run_forward()
        alb_wl *= 10. ** (-3.)

        if self.param['fit_wtr_cld'] and self.param['cld_frac'] != 1.0:
            alb = self.adjust_for_cld_frac(alb, cube)
            self.cube_to_param(cube)

        wl, model = model_finalizzation(self.param, alb_wl, alb, planet_albedo=self.param['albedo_calc'], fp_over_fs=self.param['fp_over_fs'])
        plt.plot(wl, model, color='black', linestyle='dashed', label='H$_2$O cloud')
        single_contr = np.array([wl, model]).T
        np.savetxt(self.param['out_dir'] + contr_out + 'H2O_cld.dat', single_contr)
        self.param['contribution'] = False

        self.param['spectrum']['wl'] = temp_wl + 0.0
        self.param['start_c_wl_grid'] = find_nearest(self.param['wl_C_grid'], self.param['min_wl']) - 35
        self.param['stop_c_wl_grid'] = find_nearest(self.param['wl_C_grid'], self.param['max_wl']) + 35
        plt.errorbar(self.param['spectrum']['wl'], self.param['spectrum']['Fplanet'], yerr=self.param['spectrum']['error_p'],
                     linestyle='', linewidth=0.5, color='black', marker='o', markerfacecolor='red', markersize=4, capsize=1.75, label='Data')

        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.xlabel('Wavelength [$\mu$m]')
        if self.param['albedo_calc']:
            plt.ylabel('Albedo')
        elif self.param['fp_over_fs']:
            plt.ylabel('Contrast Ratio (F$_p$/F$_{\star}$)')
        else:
            plt.ylabel('Planetary flux [W/m$^2$]')
        fig.tight_layout()

        if self.param['mol_custom_wl']:
            if solutions is None:
                plt.savefig(self.param['out_dir'] + 'Nest_mol_contribution_extended.pdf')
            else:
                plt.savefig(self.param['out_dir'] + 'Nest_mol_contribution_extended (solution ' + str(solutions) + ').pdf')
            plt.close()
        else:
            if solutions is None:
                plt.savefig(self.param['out_dir'] + 'Nest_mol_contribution.pdf')
            else:
                plt.savefig(self.param['out_dir'] + 'Nest_mol_contribution (solution ' + str(solutions) + ').pdf')
            plt.close()

        if is_bins:
            self.param['spectrum']['bins'] = True

    def plot_posteriors(self, prefix, multinest_results, parameters, mds_orig):
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

            if not self.param['fit_amm_cld']:
                z -= 3

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
                The contour levels to draw. Default are `[0.5, 1, 1.5, 2]`-sigma.

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

            # The default "sigma" contour levels.
            if levels is None:
                levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)

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

        def cornerplot(results, span=None, quantiles=[0.16, 0.5, 0.84],
                       color='black', smooth=0.02, hist_kwargs=None,
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

                n, b, _ = ax.hist(x, bins=75, weights=weights, color=color, range=np.sort(span[i]), **hist_kwargs)

                # if isinstance(sx, int_type):
                #     # If `sx` is an integer, plot a weighted histogram with
                #     # `sx` bins within the provided bounds.
                #     n, b, _ = ax.hist(x, bins=sx, weights=weights, color=color, range=np.sort(span[i]), **hist_kwargs)
                # else:
                #     # If `sx` is a float, oversample the data relative to the
                #     # smoothing filter by a factor of 10, then use a Gaussian
                #     # filter to smooth the results.
                #     bins = int(round(10. / sx))
                #     n, b = np.histogram(x, bins=bins, weights=weights, range=np.sort(span[i]))
                #     n = norm_kde(n, 10.)
                #     b0 = 0.5 * (b[1:] + b[:-1])
                #     n, b, _ = ax.hist(b0, bins=b, weights=n,
                #                       range=np.sort(span[i]), color=color,
                #                       **hist_kwargs)

                ax.set_ylim([0., max(n) * 1.05])
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
            plt.savefig(prefix + 'Nest_trace.pdf', bbox_inches='tight')
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

    def calc_spectra(self, mc_samples):
        if self.param['mol_custom_wl']:
            new_wl = np.loadtxt(self.param['pkg_dir'] + 'forward_mod/Data/wl_bins/bins_02_50_R500.dat')
            new_wl_central = np.mean(new_wl, axis=1)
            start = 0
            stop = len(new_wl_central) - 1
        else:
            new_wl = np.loadtxt(self.param['pkg_dir'] + 'forward_mod/Data/wl_bins/bins_02_20_R500.dat')
            new_wl_central = np.mean(new_wl, axis=1)
            start = find_nearest(new_wl_central, min(self.param['spectrum']['wl']) - 0.05)
            stop = find_nearest(new_wl_central, max(self.param['spectrum']['wl']) + 0.05)

        wl_len = len(self.param['spectrum']['wl'])
        if self.param['spectrum']['bins']:
            temp = np.array([self.param['spectrum']['wl_low'], self.param['spectrum']['wl_high'], self.param['spectrum']['wl']]).T
        else:
            temp = self.param['spectrum']['wl']
        self.param['spectrum']['wl'] = new_wl_central[start:stop]
        self.param['spectrum']['wl_low'] = new_wl[start:stop, 0]
        self.param['spectrum']['wl_high'] = new_wl[start:stop, 1]

        temp_min, temp_max = self.param['min_wl'] + 0.0, self.param['max_wl'] + 0.0
        self.param['min_wl'] = min(self.param['spectrum']['wl'])
        self.param['max_wl'] = max(self.param['spectrum']['wl'])
        self.param['start_c_wl_grid'] = find_nearest(self.param['wl_C_grid'], self.param['min_wl']) - 35
        self.param['stop_c_wl_grid'] = find_nearest(self.param['wl_C_grid'], self.param['max_wl']) + 35

        if mc_samples.shape[0] < self.param['n_likelihood_data']:
            self.param['n_likelihood_data'] = mc_samples.shape[0] - MPIsize
        else:
            pass

        samples = np.zeros((len(self.param['spectrum']['wl']), int(self.param['n_likelihood_data'] / MPIsize) + 1))
        samples[:, 0] = self.param['spectrum']['wl']
        loglike_data = np.zeros((int(self.param['n_likelihood_data'] / MPIsize), wl_len))

        if MPIrank == 0:
            print('\nCalculating the likelihood per data point')
            try:
                os.mkdir(self.param['out_dir'] + 'loglikelihood_per_datapoint/')
            except OSError:
                pass

        idx = np.random.choice(mc_samples.shape[0], int(self.param['n_likelihood_data']), replace=False)

        for i in range(int(self.param['n_likelihood_data'] / MPIsize)):
            cube = mc_samples[idx[i], :]
            self.cube_to_param(cube)
            mod = FORWARD_MODEL(self.param, retrieval=False, canc_metadata=True)
            alb_wl, alb = mod.run_forward()
            alb_wl *= 10. ** (-3.)

            if self.param['fit_wtr_cld'] and self.param['rocky'] and self.param['cld_frac'] != 1.0:
                alb = self.adjust_for_cld_frac(alb, cube)
                self.cube_to_param(cube)

            _, samples[:, i + 1] = model_finalizzation(self.param, alb_wl, alb, planet_albedo=self.param['albedo_calc'], fp_over_fs=self.param['fp_over_fs'])

            if self.param['spectrum']['bins']:
                model = custom_spectral_binning(temp[:, :2], self.param['spectrum']['wl'], samples[:, i + 1], bins=True)
            else:
                model = spectres(temp, self.param['spectrum']['wl'], samples[:, i + 1], fill=False)

            # Calculate likelihood per single datapoint
            chi = (self.param['spectrum']['Fplanet'] - model) / self.param['spectrum']['error_p']
            loglikelihood = ((-1.) * np.log(self.param['spectrum']['error_p'] * np.sqrt(2.0 * math.pi))) - (0.5 * chi * chi)

        np.savetxt(self.param['out_dir'] + 'loglikelihood_per_datapoint/loglike_' + str(MPIrank) + '.dat', loglike_data)
        np.savetxt(self.param['out_dir'] + 'loglikelihood_per_datapoint/samples_' + str(MPIrank) + '.dat', samples)

        if self.param['spectrum']['bins']:
            self.param['spectrum']['wl'] = temp[:, 2] + 0.0
            self.param['spectrum']['wl_low'] = temp[:, 0] + 0.0
            self.param['spectrum']['wl_high'] = temp[:, 1] + 0.0
        else:
            self.param['spectrum']['wl'] = temp + 0.0

    def adjust_for_cld_frac(self, albedo, mlnst_cube):
        self.param['fit_wtr_cld'] = False
        self.cube_to_param(mlnst_cube, free_cld_calc=True)
        mod = FORWARD_MODEL(self.param, retrieval=False, canc_metadata=True)
        _, alb_no_cld = mod.run_forward()
        self.param['fit_wtr_cld'] = True
        return (self.param['cld_frac'] * albedo) + ((1.0 - self.param['cld_frac']) * alb_no_cld)
