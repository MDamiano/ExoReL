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

                    plot_nest_spec(self, cube[:, 0], solutions=None)
                    plot_chemistry(self.param, solutions=None)
                    if self.param['surface_albedo_parameters'] > 1:
                        plot_surface_albedo(self.param, solutions=None)
                    if self.param['plot_contribution'] and self.param['obs_numb'] is None:
                        plot_contribution(self, cube[:, 0], solutions=None)
                else:
                    cube = np.ones((len(s['modes'][0]['maximum a posterior']), mds))
                    for i in range(0, mds):
                        cube[:, i] = list(s['modes'][i]['maximum a posterior'])

                        plot_nest_spec(self, cube[:, i], solutions=i + 1)
                        plot_chemistry(self.param, solutions=i + 1)
                        if self.param['surface_albedo_parameters'] > 1:
                            plot_surface_albedo(self.param, solutions=i + 1)
                        if self.param['plot_contribution'] and self.param['obs_numb'] is None:
                            plot_contribution(self, cube[:, i], solutions=i + 1)

                if self.param['spectrum']['bins']:
                    data_spec = np.array([self.param['spectrum']['wl_low'], self.param['spectrum']['wl_high'], self.param['spectrum']['wl'], self.param['spectrum']['Fplanet'], self.param['spectrum']['error_p']]).T
                else:
                    data_spec = np.array([self.param['spectrum']['wl'], self.param['spectrum']['Fplanet'], self.param['spectrum']['error_p']]).T
                np.savetxt(self.param['out_dir'] + 'data_spectrum.dat', data_spec)

            if self.param['plot_posterior']:
                # Delegate posterior plotting to centralized plotting module
                plot_posteriors(self, prefix, multinest_results, parameters, mds_orig)

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

        if self.param['double_cloud']:
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
