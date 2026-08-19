import shutil

from .__basics import *
from .__utils import *
from .__forward import (
    apply_forward_evaluation,
    forward,
    prepare_forward_parameters,
)
from .__plotting import *
from .utils.likelihood import gaussian_loglike
from .utils.prior import Mp_Rp_prior, gaussian_prior, uniform_prior
from .utils.warnings_and_errors import InvalidVMRCompositionError
from . import __version__

# trying to initiate MPI parallelization
try:
    from mpi4py import MPI

    MPIrank = MPI.COMM_WORLD.Get_rank()
    MPIsize = MPI.COMM_WORLD.Get_size()
    MPIimport = True
except ImportError:
    MPIimport = False
    MPIrank = 0
    MPIsize = 1

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
        self.param = par.copy()
        if MPIrank == 0:
            self.param = pre_load_variables(self.param)
            self.param = ranges(self.param)

    def _cube_to_evaluation(self, cube, n_obs=0):
        """Map a physical MultiNest cube to the values consumed by ``forward``."""
        evaluation = {}
        par = 0
        if self.param['fit_p0'] and self.param['gas_par_space'] != 'partial_pressure':
            evaluation['P0'] = 10.0 ** cube[par]
            par += 1

        if self.param['fit_wtr_cld'] and self.param['PT_profile_type'] == 'isothermal':
            evaluation['pH2O'] = 10.0 ** cube[par]
            evaluation['dH2O'] = 10.0 ** cube[par + 1]
            evaluation['crH2O'] = 10.0 ** cube[par + 2]
            par += 3

        if (
            self.param['fit_amm_cld']
            and self.param['PT_profile_type'] == 'isothermal'
        ):
            evaluation['pNH3'] = 10.0 ** cube[par]
            evaluation['dNH3'] = 10.0 ** cube[par + 1]
            evaluation['crNH3'] = 10.0 ** cube[par + 2]
            par += 3

        if self.param['gas_par_space'] in ('centered_log_ratio', 'clr'):
            clr_values = list(
                cube[par: par + len(self.param['fit_molecules'])]
            )
            clr_values.append(-np.sum(np.asarray(clr_values)))
            vmr_values = clr_inv(clr_values)
            for index, mol in enumerate(self.param['fit_molecules']):
                evaluation[mol] = vmr_values[index]
                par += 1
            evaluation[self.param['gas_fill']] = 1.0
            for mol in self.param['fit_molecules']:
                evaluation[self.param['gas_fill']] -= evaluation[mol]
        elif self.param['gas_par_space'] in ('volume_mixing_ratio', 'vmr'):
            fitted_total = 0.0
            for mol in self.param['fit_molecules']:
                evaluation[mol] = 10.0 ** cube[par]
                fitted_total += evaluation[mol]
                par += 1
            evaluation[self.param['gas_fill']] = 1.0 - fitted_total
        elif self.param['gas_par_space'] == 'partial_pressure':
            partial_pressures = 10.0 ** np.asarray(
                cube[par: par + len(self.param['fit_molecules'])]
            )
            evaluation['P0'] = np.sum(partial_pressures)
            for mol in self.param['fit_molecules']:
                evaluation[mol] = (10.0 ** cube[par]) / evaluation['P0']
                par += 1

        if self.param['fit_ag']:
            if self.param['surface_albedo_parameters'] == 1:
                evaluation['ag'] = cube[par] + 0.0
                par += 1
            elif self.param['surface_albedo_parameters'] == 3:
                evaluation['ag1'] = cube[par] + 0.0
                evaluation['ag2'] = cube[par + 1] + 0.0
                evaluation['ag_x1'] = cube[par + 2] + 0.0
                par += 3
            elif self.param['surface_albedo_parameters'] == 5:
                evaluation['ag1'] = cube[par] + 0.0
                evaluation['ag2'] = cube[par + 1] + 0.0
                evaluation['ag3'] = cube[par + 2] + 0.0
                evaluation['ag_x1'] = cube[par + 3] + 0.0
                evaluation['ag_x2'] = cube[par + 4] + 0.0
                par += 5

        if self.param['fit_T']:
            if self.param['PT_profile_type'] == 'isothermal':
                evaluation['Tp'] = cube[par] + 0.0
                par += 1
            elif self.param['PT_profile_type'] == 'parametric':
                evaluation['kappa_th'] = 10.0 ** cube[par]
                evaluation['gamma'] = 10.0 ** cube[par + 1]
                evaluation['beta'] = cube[par + 2] + 0.0
                par += 3
                if self.param['fit_Tint']:
                    evaluation['Tint'] = cube[par] + 0.0
                    par += 1

        if self.param['fit_cld_frac']:
            evaluation['cld_frac'] = 10.0 ** cube[par]
            par += 1

        if self.param['fit_g']:
            evaluation['gp'] = cube[par] + 0.0
            par += 1
        if self.param['fit_Mp']:
            evaluation['Mp'] = cube[par] + 0.0
            par += 1
        if self.param['fit_Rp']:
            evaluation['Rp'] = cube[par] + 0.0
            par += 1

        if self.param['fit_p_size']:
            evaluation['p_size'] = 10.0 ** cube[par]
            par += 1
        if self.param['fit_phi']:
            evaluation['phi'] = cube[par + n_obs] * math.pi / 180.0

        return evaluation

    def _forward_cube(
        self,
        cube,
        phi=None,
        n_obs=0,
        retrieval_mode=True,
    ):
        """Evaluate one retrieval cube through the public forward pipeline."""
        evaluation = self._cube_to_evaluation(cube, n_obs=n_obs)
        forward_kwargs = {
            'evaluation': evaluation,
            'phi': phi,
            'n_obs': n_obs,
            'retrieval_mode': retrieval_mode,
            'core_number': MPIrank,
            'albedo_calc': self.param['albedo_calc'],
            'fp_over_fs': self.param['fp_over_fs'],
            'canc_metadata': True,
        }

        if platform.system() == 'Darwin':
            return forward(self.param, **forward_kwargs)

        try:
            return forward(self.param, **forward_kwargs)
        except InvalidVMRCompositionError:
            raise
        except Exception:
            import traceback

            msg = (
                f"\nForward model failed on MPI rank {MPIrank}/{MPIsize}.\n"
                + traceback.format_exc()
            )
            print(msg, file=sys.stderr, flush=True)

            try:
                os.makedirs(self.param['out_dir'], exist_ok=True)
                with open(
                    self.param['out_dir']
                    + f'mpi_rank_{MPIrank}_error.log',
                    'a',
                ) as error_file:
                    error_file.write(msg + '\n')
            except Exception:
                pass

            if MPIimport:
                MPI.COMM_WORLD.Abort(1)

            raise

    def _loglike(self, cube, ndim=None, nparams=None):
        """Return the retrieval log likelihood for one physical cube."""
        try:
            if self.param['obs_numb'] is None:
                model = self._forward_cube(cube)
                return gaussian_loglike(
                    self.param['spectrum']['Fplanet'],
                    model,
                    self.param['spectrum']['error_p'],
                )

            loglikelihood = 0.0
            for obs in range(self.param['obs_numb']):
                phi = (
                    None
                    if self.param['fit_phi']
                    else self.param['phi' + str(obs)]
                )
                model = self._forward_cube(
                    cube,
                    phi=phi,
                    n_obs=obs,
                )
                observation = self.param['spectrum'][str(obs)]
                loglikelihood += gaussian_loglike(
                    observation['Fplanet'],
                    model,
                    observation['error_p'],
                )
            return loglikelihood
        except InvalidVMRCompositionError:
            return -1e99

    def run_retrieval(self):
        is_root = (not MPIimport) or MPIrank == 0

        if is_root:
            os.makedirs(self.param['out_dir'], exist_ok=True)
            copy_parfile_to_output(self.param)

            print(f"Running ExoReL – version {__version__}")
            # check if the run is done, in case clean c meta files
            if not os.path.isfile(self.param['out_dir'] + self.param['name_p'] + '_params.json'):
                clean_c_files(self.param['pkg_dir'])
            if self.param['rocky']:
                print('Using ExoReL-R for small planets')
                if self.param['gas_par_space'] in ('clr', 'centered_log_ratio'):
                    print('Using modified priors: ' + str(self.param['mod_prior']))
            else:
                print('Using ExoReL-R for giant gaseous planets')

        parameters, n_params = retrieval_par_and_npar(self.param)
        self.param['fitting_parameters'] = parameters
        if self.param['gas_par_space'] in ('clr', 'centered_log_ratio') and self.param['mod_prior']:
            ppf = np.loadtxt(self.param['pkg_dir'] + 'forward_mod/Data/prior/prior_cube_' + str(len(self.param['fit_molecules'])) + 'gas.dat')

        if self.param['physics_model'] == 'dataset':
            # Update retrieval ranges to dataset min/max and validate coverage
            if is_root:
                self.param = adjust_ranges_from_dataset(self.param)

        if MPIimport and MPIsize > 1:
            MPI.COMM_WORLD.Barrier()  # wait for everybody to synchronize here
            self.param = MPI.COMM_WORLD.bcast(self.param, root=0)
            MPI.COMM_WORLD.Barrier()  # wait for everybody to synchronize here

        def prior(cube, ndim, nparams):
            par = 0
            if self.param['fit_p0'] and self.param['gas_par_space'] != 'partial_pressure':
                cube[par] = uniform_prior(self.param, 'p0', cube[par])  # uniform prior between   3  :  11    -> P0, surface pressure
                par += 1
            if self.param['fit_wtr_cld'] and self.param['PT_profile_type'] == 'isothermal':
                cube[par] = uniform_prior(self.param, 'ptopw', cube[par])  # uniform prior between   0  :  8     -> P H2O cloud top [Pa]
                cube[par + 1] = uniform_prior(self.param, 'dcldw', cube[par + 1])  # uniform prior between   0  :  8.5   -> D H2O cloud [Pa]
                cube[par + 2] = uniform_prior(self.param, 'crh2o', cube[par + 2])  # uniform prior between -12  :  0     -> CR H2O
                par += 3

            if self.param['fit_amm_cld'] and self.param['PT_profile_type'] == 'isothermal':
                cube[par] = uniform_prior(self.param, 'ptopa', cube[par])  # uniform prior between   0  :  8     -> P NH3 cloud top [Pa]
                cube[par + 1] = uniform_prior(self.param, 'dclda', cube[par + 1])  # uniform prior between   0  :  8.5   -> D H2O cloud [Pa]
                cube[par + 2] = uniform_prior(self.param, 'crnh3', cube[par + 2])  # uniform prior between -12  :  0     -> CR NH3
                par += 3

            for mol in self.param['fit_molecules']:
                if self.param['gas_par_space'] in ('centered_log_ratio', 'clr'):
                    if self.param['mod_prior']:
                        cube[par] = ppf[find_nearest(ppf[:, 0], cube[par]), 1]  # modified prior for clr
                    else:
                        cube[par] = uniform_prior(self.param, 'clr' + mol, cube[par])  # uniform clr prior between -25 : 25
                elif self.param['gas_par_space'] in ('volume_mixing_ratio', 'vmr'):
                    cube[par] = uniform_prior(self.param, 'vmr' + mol, cube[par])  # uniform vmr prior between -12 : 0
                elif self.param['gas_par_space'] == 'partial_pressure':
                    cube[par] = uniform_prior(self.param, 'pp' + mol, cube[par])  # uniform partial pressure prior between -10 : 10
                par += 1

            if self.param['fit_ag']:
                if self.param['surface_albedo_parameters'] == 1:
                    cube[par] = uniform_prior(self.param, 'ag', cube[par])  # uniform prior between   0.0  :  0.5     -> Ag, surface albedo
                    par += 1
                elif self.param['surface_albedo_parameters'] == 3:
                    for surf_alb in [1, 2]:
                        cube[par + (surf_alb - 1)] = uniform_prior(self.param, 'ag' + str(surf_alb), cube[par + (surf_alb - 1)])
                    cube[par + surf_alb] = uniform_prior(self.param, 'ag_x1', cube[par + surf_alb])
                    par += 3
                elif self.param['surface_albedo_parameters'] == 5:
                    for surf_alb in [1, 2, 3]:
                        cube[par + (surf_alb - 1)] = uniform_prior(self.param, 'ag' + str(surf_alb), cube[par + (surf_alb - 1)])
                    cube[par + surf_alb] = uniform_prior(self.param, 'ag_x1', cube[par + surf_alb])
                    cube[par + surf_alb + 1] = cube[par + surf_alb] + uniform_prior(self.param, 'ag_x2', cube[par + surf_alb + 1])
                    par += 5

            if self.param['fit_T']:
                if self.param['PT_profile_type'] == 'isothermal':
                    cube[par] = uniform_prior(self.param, 'tp', cube[par])  # uniform prior between   0  :  700   -> Tp, planetary temperature
                    par += 1
                elif self.param['PT_profile_type'] == 'parametric':
                    cube[par] = uniform_prior(self.param, 'kappa_th', cube[par])  # log uniform prior between 1e-10 : 1e0 -> kappa_th, thermal radiation opacity
                    cube[par + 1] = uniform_prior(self.param, 'gamma', cube[par + 1])  # log uniform prior between   1e-10 : 1e10   -> gamma, ratio visible opacity : kappa_th
                    cube[par + 2] = uniform_prior(self.param, 'beta', cube[par + 2])  # uniform prior between   0 : 2   -> beta, scaling factor for T_equilibrium
                    par += 3
                    if self.param['fit_Tint']:
                        cube[par] = uniform_prior(self.param, 'Tint', cube[par])  # uniform prior between   0 : 300   -> Tint, internal temperature
                        par += 1

            if self.param['fit_cld_frac']:
                cube[par] = uniform_prior(self.param, 'cld_frac', cube[par])  # uniform prior between   -3.0  :  0.0     -> Log(clf_frac), cloud fraction
                par += 1

            if self.param['fit_g']:
                cube[par] = uniform_prior(self.param, 'gp', cube[par])  # uniform prior between   1.0  :  6   -> g [cm/s2]
                par += 1

            if self.param['fit_Mp'] and self.param['fit_Rp']:
                if self.param['Rp_prior_type'] != 'M_R_prior' and self.param['Mp_prior_type'] != 'M_R_prior':
                    cube[par] = Mp_Rp_prior(self.param, 'Mp', cube[par])  # Mass prior - independent or gaussian
                    cube[par + 1] = Mp_Rp_prior(self.param, 'Rp', cube[par + 1])  # Radius prior - independent or gaussian
                    par += 2
                elif self.param['Rp_prior_type'] == 'M_R_prior' and self.param['Mp_prior_type'] != 'M_R_prior':
                    cube[par] = Mp_Rp_prior(self.param, 'Mp', cube[par])  # Mass prior - independent or gaussian
                    cube[par + 1] = Mp_Rp_prior(self.param, 'Rp', cube[par + 1], mp_value=cube[par])  # Radius prior - 2D prior
                    par += 2
                elif self.param['Rp_prior_type'] != 'M_R_prior' and self.param['Mp_prior_type'] == 'M_R_prior':
                    cube[par + 1] = Mp_Rp_prior(self.param, 'Rp', cube[par + 1])  # Radius prior - independent or gaussian
                    cube[par] = Mp_Rp_prior(self.param, 'Mp', cube[par], rp_value=cube[par + 1])  # Mass prior - 2D prior
                    par += 2
            elif self.param['fit_Mp'] and not self.param['fit_Rp']:
                cube[par] = Mp_Rp_prior(self.param, 'Mp', cube[par])  # Mass prior
                par += 1
            elif self.param['fit_Rp'] and not self.param['fit_Mp']:
                cube[par] = Mp_Rp_prior(self.param, 'Rp', cube[par])  # Radius prior
                par += 1

            if self.param['fit_p_size']:
                cube[par] = uniform_prior(self.param, 'p_size', cube[par])  # Particle size uniform prior
                par += 1

            if self.param['fit_phi']:
                if self.param['obs_numb'] is None:
                    if self.param['phi_err'] is None: 
                        cube[par] = uniform_prior(self.param, 'phi', cube[par])  # uniform prior between   0  :  180   -> deg
                    else:
                        cube[par] = gaussian_prior(self.param, 'phi', cube[par])  # gaussian prior -> phi, phase angle
                    par += 1
                else:
                    for _ in range(0, self.param['obs_numb']):
                        cube[par] = uniform_prior(self.param, 'phi', cube[par])  # uniform prior between   0  :  180   -> deg
                        par += 1

        if is_root:
            time1 = time.time()

        pymultinest.run(LogLikelihood=self._loglike,
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

        if is_root:  # Plot Nest_spectrum
            time2 = time.time()
            elapsed((time2 - time1) * (10 ** 9))

        prefix = self.param['out_dir'] + self.param['name_p'] + '_'
        if is_root:
            json.dump(parameters, open(prefix + 'params.json', 'w'))  # save parameter names

        ### POST-PROCESSING ###
        self.param['model_n_par'] = len(parameters)
        multinest_results = pymultinest.Analyzer(n_params=self.param['model_n_par'], outputfiles_basename=prefix, verbose=False)

        if is_root:
            if self.param['filter_multi_solutions']:
                s, mds = self.filter_pymultinest_modes(multinest_results)
                mds_orig = len(multinest_results.get_stats()['modes'])
            else:
                s = multinest_results.get_stats()
                mds_orig = mds = len(s['modes'])
            
            if self.param['multimodal'] and mds_orig > 1:
                self.store_nest_solutions(prefix)

        if MPIimport and MPIsize > 1:
            MPI.COMM_WORLD.Barrier()  # wait for everybody to synchronize here
            mds_orig = MPI.COMM_WORLD.bcast(mds_orig if MPIrank == 0 else None, root=0)

        check_files = os.path.exists(
            self.param['out_dir'] + 'loglike_per_datapoint_sol0.dat'
        )

        if self.param['calc_likelihood_data'] and not check_files:
            self.calc_spectra(prefix, mds_orig)

            if MPIimport and MPIsize > 1:
                MPI.COMM_WORLD.Barrier()  # wait for everybody to synchronize here

            if is_root:
                loglike_dir = []
                for mode_idx in range(0, mds_orig):
                    loglike_dir.append(self.param['out_dir'] + f'loglikelihood_per_datapoint_sol{mode_idx}/')
                for idx, folder in enumerate(loglike_dir):
                    rank_0 = np.loadtxt(folder + 'loglike_0.dat', ndmin=2)
                    rank_0_spec = np.loadtxt(folder + 'samples_0.dat', ndmin=2)
                    rank_0_par = np.loadtxt(folder + 'samples_par_0.dat', ndmin=2)
                    if self.param['fit_T'] and self.param['PT_profile_type'] == 'parametric':
                        rank_0_temp = np.loadtxt(folder + 'temp_samples_0.dat', ndmin=2)
                    for i in range(1, MPIsize):
                        rank_n = np.loadtxt(folder + f'loglike_{i}.dat', ndmin=2)
                        rank_n_spec = np.loadtxt(folder + f'samples_{i}.dat', ndmin=2)
                        rank_n_par = np.loadtxt(folder + f'samples_par_{i}.dat', ndmin=2)
                        rank_0 = np.concatenate((rank_0, rank_n), axis=0)
                        rank_0_spec = np.concatenate((rank_0_spec, rank_n_spec[:, 1:]), axis=1)
                        rank_0_par = np.concatenate((rank_0_par, rank_n_par), axis=0)
                        if self.param['fit_T'] and self.param['PT_profile_type'] == 'parametric':
                            rank_n_temp = np.loadtxt(folder + f'temp_samples_{i}.dat', ndmin=2)
                            rank_0_temp = np.concatenate((rank_0_temp, rank_n_temp[:, 1:]), axis=1)
                    np.savetxt(self.param['out_dir'] + f'loglike_per_datapoint_sol{idx}.dat', rank_0)
                    np.savetxt(self.param['out_dir'] + f'random_samples_sol{idx}.dat', rank_0_spec)
                    np.savetxt(self.param['out_dir'] + f'parameters_samples_sol{idx}.dat', rank_0_par)
                    if self.param['fit_T'] and self.param['PT_profile_type'] == 'parametric':
                        np.savetxt(self.param['out_dir'] + f'random_temp_samples_sol{idx}.dat', rank_0_temp)
                        del rank_0_temp
                    shutil.rmtree(folder)

                    self.param['spec_sample'] = rank_0_spec + 0.0
                    del rank_0, rank_0_spec, rank_0_par
        elif self.param['calc_likelihood_data'] and check_files:
            if is_root:
                print('\n"loglike_per_datapoint" files already exist. Skipping likelihood per data point calculation.')

        if is_root:
            if self.param['spectrum']['bins']:
                data_spec = np.array([self.param['spectrum']['wl_low'], self.param['spectrum']['wl_high'], self.param['spectrum']['wl'], self.param['spectrum']['Fplanet'], self.param['spectrum']['error_p']]).T
            else:
                data_spec = np.array([self.param['spectrum']['wl'], self.param['spectrum']['Fplanet'], self.param['spectrum']['error_p']]).T
            np.savetxt(self.param['out_dir'] + 'data_spectrum.dat', data_spec)

            ### PRODUCE PLOTS FROM HERE ###
            cube = np.ones((len(s['modes'][0]['maximum a posterior']), mds))
            for i in range(0, mds):
                cube[:, i] = list(s['modes'][i]['maximum a posterior'])

                if self.param['plot_models']:
                    print(f"\nPlotting solution {i + 1} of {mds} with log-evidence: {s['modes'][i]['local log-evidence']:.2f}\n")

                    plot_nest_spec(self, cube[:, i], solutions=i)
                    plot_chemistry(self.param, solutions=i)
                    if self.param['rocky']:
                        plot_mass_radius(self, cube[:, i], solutions=i, sigma=s['modes'][i]['sigma'])
                    if self.param['surface_albedo_parameters'] > 1:
                        plot_surface_albedo(self.param, solutions=i, sigma=s['modes'][i]['sigma'])

                    if os.path.exists(self.param['out_dir'] + f'random_temp_samples_sol{i}.dat') and self.param['fit_T'] and self.param['PT_profile_type'] == 'parametric':
                        plot_PT_profile(self, cube[:, i], solutions=i)
                    elif self.param['fit_T'] and self.param['PT_profile_type'] == 'parametric':
                        print('\nTo plot P-T profiles, the calculation of the temperatures samples must be enabled (calc_likelihood_data = True).')

                    if self.param['obs_numb'] is None:
                        plot_contribution(self, cube[:, i], solutions=i)

                    if os.path.exists(self.param['out_dir'] + f'loglike_per_datapoint_sol{i}.dat') and os.path.exists(self.param['out_dir'] + f'parameters_samples_sol{i}.dat') and self.param['plot_elpd_stats']:
                        elpd_loo_stats(self, parameters, solutions=i)
                    else:
                        if self.param['plot_elpd_stats']:
                            print('\nTo plot elpd statistics, the calculation of the likelihood per data point must be enabled (calc_likelihood_data = True).')

            if self.param['plot_posterior']:
                # Delegate posterior plotting to centralized plotting module
                plot_posteriors(self, prefix, multinest_results, parameters, mds_orig)

        if is_root:
            write_stats_summary_files(self.param, prefix, multinest_results.get_stats(), len(parameters))

        if MPIimport:
            MPI.Finalize()


    def filter_pymultinest_modes(self, mres):
        mds = len(mres.get_stats()['modes'])
        if mds == 1:
            return mres.get_stats(), mds

        max_ev, max_idx = -np.inf, 0
        for i in range(mds):
            if mres.get_stats()['modes'][i]['local log-evidence'] > max_ev:
                max_ev = mres.get_stats()['modes'][i]['local log-evidence'] + 0.0
                max_idx = i

        filtered_modes = [mres.get_stats()['modes'][max_idx]]
        filtered_modes[-1]['index'] = len(filtered_modes) - 1
        for i in range(mds):
            if (
                i != max_idx
                and max_ev - mres.get_stats()['modes'][i]['local log-evidence']
                < MULTI_SOLUTION_DELTA_LOG_EVIDENCE_THRESHOLD
            ):
                filtered_modes.append(mres.get_stats()['modes'][i])
                filtered_modes[-1]['index'] = len(filtered_modes) - 1

        s = mres.get_stats()
        s['modes'] = filtered_modes

        filtered_count = mds - len(s['modes'])
        if filtered_count > 0:
            print(
                f"\n{filtered_count} modes have been filtered due to low "
                "significance"
            )
        return s, len(s['modes'])


    def cube_to_param(self, cube, n_obs=0):
        """Prepare plotting/post-processing parameters from a retrieval cube."""
        evaluation = self._cube_to_evaluation(cube, n_obs=n_obs)
        self.param = prepare_forward_parameters(
            self.param,
            evaluation=evaluation,
            core_number=None,
        )


    def calc_spectra(self, prefix, mds_orig):
        """Draw posterior samples and evaluate them on the observed grid."""
        def sample_from_weighted_distr(x, w, m, rng=None):
            rng = np.random.default_rng() if rng is None else rng
            x = np.asarray(x)
            w = np.asarray(w, dtype=float)
            if m <= 0:
                raise ValueError("'n_likelihood_data' must be a positive integer.")
            if not np.all(np.isfinite(w)) or np.any(w < 0.0):
                raise ValueError('Posterior weights must be finite and non-negative.')
            weight_sum = np.sum(w)
            if not np.isfinite(weight_sum) or weight_sum <= 0.0:
                raise ValueError('Posterior weights must have a positive finite sum.')
            p = w / weight_sum
            idx = rng.choice(len(x), size=m, replace=True, p=p)
            return x[idx, :]

        if MPIrank == 0:
            print('\nCalculating the likelihood per data point')

        input_wavelength = np.asarray(
            self.param['spectrum']['wl'],
            dtype=float,
        )
        wl_len = len(input_wavelength)
        n_samples = int(self.param['n_likelihood_data'])
        if n_samples <= 0:
            raise ValueError("'n_likelihood_data' must be a positive integer.")
        if n_samples < MPIsize:
            raise ValueError(
                "'n_likelihood_data' must be at least the number of MPI ranks."
            )

        distributions = []

        if mds_orig > 1:
            for i in range(0, mds_orig):
                distributions.append(np.loadtxt(prefix + f'solution{i}.txt'))
        else:
            distributions.append(np.loadtxt(prefix + '.txt'))

        for mds in range(len(distributions)):
            mc_samples = np.atleast_2d(distributions[mds])
            if mc_samples.shape[1] < 3:
                raise ValueError(
                    'Posterior sample files must contain weight, likelihood, '
                    'and at least one fitted parameter.'
                )

            loglike_dir = self.param['out_dir'] + f'loglikelihood_per_datapoint_sol{mds}/'

            os.makedirs(loglike_dir, exist_ok=True)

            sample_par = sample_from_weighted_distr(
                mc_samples[:, 2:],
                mc_samples[:, 0],
                m=n_samples,
                rng=np.random.default_rng(42),
            )

            base_per_rank, extra = divmod(n_samples, MPIsize)
            local_n = base_per_rank + (1 if MPIrank < extra else 0)
            start_idx = MPIrank * base_per_rank + min(MPIrank, extra)
            stop_idx = start_idx + local_n

            samples = np.zeros((wl_len, local_n + 1))
            samples[:, 0] = input_wavelength
            loglike_data = np.zeros((local_n, wl_len))

            if self.param['fit_T'] and self.param['PT_profile_type'] == 'parametric':
                temp_samples = np.full((len(self.param['P_standard']) + 2, local_n + 1), np.nan)
                temp_samples[2:, 0] = self.param['P_standard']

            for i, sample_idx in enumerate(range(start_idx, stop_idx)):
                cube = sample_par[sample_idx, :]
                model_wavelength, model = self._forward_cube(
                    cube,
                    retrieval_mode=False,
                )
                if not np.array_equal(model_wavelength, input_wavelength):
                    raise ValueError(
                        'Posterior spectrum wavelength grid does not match '
                        'the input spectrum grid.'
                    )
                samples[:, i + 1] = model

                # Calculate likelihood per single datapoint
                chi = (self.param['spectrum']['Fplanet'] - model) / self.param['spectrum']['error_p']
                loglike_data[i, :] = ((-1.) * np.log(self.param['spectrum']['error_p'] * np.sqrt(2.0 * math.pi))) - (0.5 * chi * chi)

                # Calculate temperature profile
                if self.param['fit_T'] and self.param['PT_profile_type'] == 'parametric':
                    evaluation = self._cube_to_evaluation(cube)
                    temp_param = apply_forward_evaluation(
                        self.param,
                        evaluation=evaluation,
                        core_number=None,
                    )
                    temp_param['P'] = 10. ** np.arange(
                        0.0,
                        np.log10(temp_param['P0']) + 0.01,
                        step=0.01,
                    )
                    T = temp_profile(temp_param)
                    temp_samples[2:len(T)+2, i+1] = T
                    temp_samples[0, i+1] = temp_param['P'][-1]
                    temp_samples[1, i+1] = T[-1]

            np.savetxt(loglike_dir + 'loglike_' + str(MPIrank) + '.dat', loglike_data)
            np.savetxt(loglike_dir + 'samples_' + str(MPIrank) + '.dat', samples)
            np.savetxt(loglike_dir + 'samples_par_' + str(MPIrank) + '.dat', sample_par[start_idx:stop_idx, :])
            if self.param['fit_T'] and self.param['PT_profile_type'] == 'parametric':
                np.savetxt(loglike_dir + 'temp_samples_' + str(MPIrank) + '.dat', temp_samples)


    def store_nest_solutions(self, prefix):
        modes = []
        modes_weights = []
        modes_loglike = []
        chains = []
        chains_weights = []
        chains_loglike = []

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

        for nmode in range(len(modes)):
            fl = np.ones((len(np.asarray(modes_weights[nmode])), len(modes_array[nmode][0, :]) + 2))
            fl[:, 0] = np.asarray(modes_weights[nmode])
            fl[:, 1] = np.asarray(modes_loglike[nmode])
            fl[:, 2:] = modes_array[nmode]
            np.savetxt(prefix + 'solution' + str(nmode) + '.txt', fl)
