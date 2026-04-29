from .__basics import *
from .__utils import *
from .__forward import *
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
    print('CAUTION - MPI disabled. Check mpi4py installation if you want to use MPI. The work will be done on a single core')
    MPIrank = 0
    MPIsize = 1


DATASET_RANGE_ALIASES = {
    "p0": "P0",
    "ptopw": "Pw_top",
    "dcldw": "cldw_depth",
    "crh2o": "CR_H2O",
    "ptopa": "Pa_top",
    "dclda": "clda_depth",
    "crnh3": "CR_NH3",
    "tp": "Tp",
    "ag": "Ag",
    "ag1": "Ag1",
    "ag2": "Ag2",
    "ag3": "Ag3",
    "ag_x1": "Ag_x1",
    "ag_x2": "Ag_x2",
}

PLANET_CLASS_MASS_THRESHOLD_MJ = 0.06


def dataset_column_name(name):
    if name.endswith('_range'):
        name = name[:-6]
    return DATASET_RANGE_ALIASES.get(name, name)


def dataset_random_seed(param):
    seed = param.get('random_seed')
    if seed is None:
        seed = param.get('seed')
    if seed is None:
        return None
    try:
        return int(seed)
    except (TypeError, ValueError) as exc:
        raise ValueError("'random_seed' must be an integer.") from exc


def mixed_planet_class_enabled(param):
    explicit = param.get('mixed_planet_class')
    if explicit is not None:
        return bool(explicit)

    mp_range = param.get('Mp_range')
    if mp_range is None:
        return False

    lo, hi = sorted([float(mp_range[0]), float(mp_range[1])])
    return lo < PLANET_CLASS_MASS_THRESHOLD_MJ <= hi


def configure_mixed_planet_class(param):
    param['mixed_planet_class'] = True
    param['gaseous_planet'] = False
    param['rocky'] = True
    param['gas_par_space'] = 'vmr'
    param['gas_fill'] = 'H2'
    param['fit_H2'] = False
    if 'fit_molecules' in param:
        param['fit_molecules'] = [
            mol for mol in param['fit_molecules'] if param.get('fit_' + mol, False)
        ]

    has_vmr_bounds = 'vmr_range' in param
    for mol in param.get('supported_molecules', []):
        has_vmr_bounds = has_vmr_bounds or ('vmr' + mol + '_range' in param)
    if not has_vmr_bounds:
        param['vmr_range'] = [-12.0, 0.0]

    if param.get('P0') is None and 'P0_range' not in param and 'p0_range' not in param:
        param['P0'] = 1.0e5

    return param


def vmr_filler_column(param):
    if param.get('gas_par_space') in ('volume_mixing_ratio', 'vmr') and param.get('gas_fill') is not None:
        return str(param['gas_fill']) + '_filler'
    return None


def dataset_header_columns(par, include_planet_class=False, filler_column=None):
    columns = ['index']
    if include_planet_class:
        columns.append('is_gaseous')
    columns.extend([dataset_column_name(str(p)) for p in par])
    if filler_column is not None:
        columns.append(filler_column)
    return columns


class GEN_DATASET:
    def __init__(self, par):
        self.param = copy.deepcopy(par)
        if self.param['PT_profile_type'] == 'isothermal' and (
            'Tp_range' in self.param or 'tp_range' in self.param
        ):
            self.param['fit_T'] = True
        if mixed_planet_class_enabled(self.param):
            self.param = configure_mixed_planet_class(self.param)
        self.param = pre_load_variables(self.param)

    def run(self):
        if MPIrank == 0:
            print(f"Running ExoReL – version {__version__}")
            npar, par = detect_gen_npar(self.param)
            random_seed = dataset_random_seed(self.param)
        
            # Generate unit-hypercube samples X in [0,1)^npar
            if self.param['optimizer'] == 'sobol':
                sampler = sp.stats.qmc.Sobol(d=npar, scramble=True, seed=random_seed)
                # Best practice: draw 2**m points
                X = sampler.random_base2(m=int(np.ceil(np.log2(self.param['n_spectra']))))
            
            else:
                if random_seed is None:
                    X = np.random.random((self.param['n_spectra'], npar))
                else:
                    X = np.random.RandomState(random_seed).random_sample((self.param['n_spectra'], npar))

            # Scale samples to parameter bounds using ranges in self.param
            lower, upper = [], []
            gps = self.param.get('gas_par_space')
            for key in par:
                # If key already encodes a range (e.g., 'p0_range'), use it directly
                if key.endswith('_range') and key in self.param:
                    rng = self.param[key]
                else:
                    # Prefer explicit '<key>_range'
                    rng = self.param.get(key + '_range')
                    if rng is None:
                        # Handle molecule keys based on gas parameter space
                        if gps in ('centered_log_ratio', 'clr'):
                            rng = self.param.get('clr' + key + '_range', self.param.get('clr_range'))
                        elif gps in ('volume_mixing_ratio', 'vmr'):
                            rng = self.param.get('vmr' + key + '_range', self.param.get('vmr_range'))
                        elif gps == 'partial_pressure':
                            rng = self.param.get('pp' + key + '_range', self.param.get('pp_range'))
                if rng is None:
                    raise KeyError(f"Range for parameter '{key}' not found in param.")
                lower.append(rng[0])
                upper.append(rng[1])

            # Structured mapping of X columns to parameter names and bounds
            include_planet_class = bool(self.param.get('mixed_planet_class', False))
            filler_column = vmr_filler_column(self.param)
            csv_columns = [dataset_column_name(str(p)) for p in par]
            lines = [f"[{i:02d}] {name}: {lo} -> {hi}" for i, (name, lo, hi) in enumerate(zip(csv_columns, lower, upper))]
            print(f"Synthesizing {X.shape[0]} spectra using {npar} sampled parameters.")
            if random_seed is not None:
                print(f"Dataset random seed: {random_seed}")
            if include_planet_class:
                print(f"Mixed planet classes enabled: Mp < {PLANET_CLASS_MASS_THRESHOLD_MJ} Mj is rocky, otherwise gaseous.")
            print("Columns mapping (index, parameter, lower, upper):\n" + "\n".join(lines))
            X_scaled = sp.stats.qmc.scale(X, lower, upper)

            # Build the design matrix header (rank 0 only). Actual CSV write happens later.
            header = ','.join(dataset_header_columns(par, include_planet_class, filler_column))

        # Synchronize all processes, then broadcast X_scaled from rank 0
        if MPIsize > 1:
            MPI.COMM_WORLD.Barrier()
            X_scaled = MPI.COMM_WORLD.bcast(X_scaled if MPIrank == 0 else None, root=0)
            par = MPI.COMM_WORLD.bcast(par if MPIrank == 0 else None, root=0)

        # Check for existing dataset and prepare appending/indexing strategy
        # - Ensure output folder exists.
        # - If dataset.csv exists, validate column count, determine next index, and append.
        # - Otherwise, create a new dataset.csv with header.
        csv_path = os.path.join(self.param['out_dir'], 'dataset.csv')

        # Rank 0 performs filesystem I/O; broadcasts decisions
        if MPIrank == 0:
            os.makedirs(self.param['out_dir'], exist_ok=True)

            append_mode = False
            start_index = 0

            # Validate that the existing header matches exactly the expected names
            if os.path.isfile(csv_path):
                with open(csv_path, 'r') as f:
                    first_line = f.readline().strip()
                header_cols = [h.strip() for h in first_line.split(',')]
                expected_cols = dataset_header_columns(
                    par,
                    bool(self.param.get('mixed_planet_class', False)),
                    vmr_filler_column(self.param),
                )
                if header_cols != expected_cols:
                    raise ValueError('Existing dataset.csv header does not match expected columns: ' + ','.join(header_cols))

                # Determine max index to continue numbering
                try:
                    # Fast path: load first column only
                    existing_idx = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=0)
                    if existing_idx.size == 0:
                        start_index = 0
                    else:
                        # np.loadtxt returns scalar for single row; handle both cases
                        start_index = int(np.max(existing_idx)) + 1
                except Exception:
                    # Fallback: stream through file
                    max_idx = -1
                    with open(csv_path, 'r') as f:
                        _ = f.readline()  # skip header
                        for line in f:
                            if not line:
                                continue
                            try:
                                val = int(line.split(',')[0])
                                if val > max_idx:
                                    max_idx = val
                            except Exception:
                                continue
                    start_index = max_idx + 1 if max_idx >= 0 else 0

                append_mode = True

            # Write/append the design matrix built from X_scaled
            nrows = X_scaled.shape[0]
            idx = (np.arange(nrows, dtype=np.int64) + start_index).reshape(-1, 1)
            csv_param_columns = [dataset_column_name(str(p)) for p in par]
            filler_column = vmr_filler_column(self.param)
            filler_values = None
            if filler_column is not None:
                sampled_gas_columns = [
                    csv_param_columns.index(mol)
                    for mol in self.param['fit_molecules']
                    if mol in csv_param_columns and mol != self.param['gas_fill']
                ]
                if sampled_gas_columns:
                    filler_values = 1.0 - np.sum(10.0 ** X_scaled[:, sampled_gas_columns], axis=1)
                else:
                    filler_values = np.ones(nrows)
                filler_values = np.maximum(0.0, filler_values).reshape(-1, 1)
                with np.errstate(divide='ignore'):
                    filler_values = np.log10(filler_values)

            if self.param.get('mixed_planet_class', False):
                csv_values = X_scaled.copy()
                try:
                    mp_idx = csv_param_columns.index('Mp')
                except ValueError as exc:
                    raise ValueError('Mixed rocky/gaseous datasets require Mp to be sampled with Mp_range.') from exc
                is_gaseous_col = (
                    csv_values[:, mp_idx] >= PLANET_CLASS_MASS_THRESHOLD_MJ
                ).astype(np.int64).reshape(-1, 1)
                for surface_col in ['Ag', 'Ag1', 'Ag2', 'Ag3']:
                    if surface_col in csv_param_columns:
                        csv_values[is_gaseous_col[:, 0] == 1, csv_param_columns.index(surface_col)] = 0.0
                data = np.concatenate((idx, is_gaseous_col, csv_values), axis=1)
            else:
                data = np.concatenate((idx, X_scaled), axis=1)
            if filler_values is not None:
                data = np.concatenate((data, filler_values), axis=1)

            if append_mode:
                with open(csv_path, 'ab') as f:
                    np.savetxt(f, data, delimiter=',', comments='', fmt='%.17g')
            else:
                header = ','.join(dataset_header_columns(
                    par,
                    bool(self.param.get('mixed_planet_class', False)),
                    vmr_filler_column(self.param),
                ))
                np.savetxt(csv_path, data, delimiter=',', header=header, comments='', fmt='%.17g')

            # Update sidecar metadata
            meta = {
                'columns': dataset_header_columns(
                    par,
                    bool(self.param.get('mixed_planet_class', False)),
                    vmr_filler_column(self.param),
                ),
                'n_rows': int(start_index + nrows),
                'n_cols': int(data.shape[1] - 1),
                'mixed_planet_class': bool(self.param.get('mixed_planet_class', False)),
                'planet_class_mass_threshold_Mj': PLANET_CLASS_MASS_THRESHOLD_MJ if self.param.get('mixed_planet_class', False) else None,
                'random_seed': dataset_random_seed(self.param),
                'ExoReL_version': str(__version__),
            }
            with open(os.path.join(self.param['out_dir'], 'dataset_meta.json'), 'w') as f:
                json.dump(meta, f, separators=(',', ':'))

        if MPIsize > 1:
            MPI.COMM_WORLD.Barrier()
            # Broadcast starting index to all ranks
            start_index = MPI.COMM_WORLD.bcast(start_index if MPIrank == 0 else None, root=0)
        
        # split X_scaled among MPI ranks for parallel processing if MPIsize is > 0.
        # Loop over the subset of X_scaled assigned to each rank, assign the samples
        # to the parameters in the self.param dictionary and generate the spectra.
        # Recompute the ordered parameter list consistently across ranks

        if MPIimport or MPIrank == 0:
            # Determine work partition for this rank
            total = X_scaled.shape[0]
            size = MPIsize if MPIimport else 1
            rank = MPIrank if MPIimport else 0
            start = (total * rank) // size
            end = (total * (rank + 1)) // size

            gps = self.param.get('gas_par_space')

            other_params = ["Rs", "Ms", "Ts", "major-a", "Tp", "Ag", "Ag1", "Ag2", "Ag3", "Ag_x1", "Ag_x2", "phi"]

            copy_tp = copy.deepcopy(self.param['Tp'])
            copy_phi = copy.deepcopy(self.param['phi'])
            base_class_values = {
                key: copy.deepcopy(self.param.get(key))
                for key in [
                    'P0', 'rocky', 'gaseous_planet', 'fit_p0', 'fit_ag',
                    'Ag', 'Ag1', 'Ag2', 'Ag3', 'Ag_x1', 'Ag_x2',
                    'gas_fill', 'fit_H2',
                ]
            }

            # Tight loop over this rank's chunk
            for i in range(start, end):
                row = X_scaled[i]

                # Build evaluation dict mapping each parameter to its sampled value.
                # Dataset files historically used a mix of retrieval aliases
                # (e.g. ptopw_range) and forward-model names (e.g. Pw_top_range).
                eval_map = {}
                for j, name in enumerate(par):
                    if name.endswith('_range'):
                        key = name[:-6]
                    else:
                        # Molecule names and other direct keys
                        key = name
                    key = DATASET_RANGE_ALIASES.get(key, key)
                    eval_map[key] = row[j]
                
                if self.param['fit_p0'] and gps != 'partial_pressure' and 'P0' in eval_map:
                    self.param['P0'] = 10. ** eval_map['P0']

                if self.param['fit_wtr_cld'] and self.param['PT_profile_type'] == 'isothermal':
                    if all(k in eval_map for k in ['Pw_top', 'cldw_depth', 'CR_H2O']):
                        for j in ['Pw_top', 'cldw_depth', 'CR_H2O']:
                            self.param[j] = 10. ** eval_map[j]
                    elif all(k in self.param for k in ['pH2O', 'dH2O', 'crH2O']):
                        self.param['Pw_top'] = 10. ** self.param['pH2O']
                        self.param['cldw_depth'] = 10. ** self.param['dH2O']
                        self.param['CR_H2O'] = 10. ** self.param['crH2O']
                if self.param['fit_amm_cld'] and self.param['PT_profile_type'] == 'isothermal':
                    if all(k in eval_map for k in ['Pa_top', 'clda_depth', 'CR_NH3']):
                        for j in ['Pa_top', 'clda_depth', 'CR_NH3']:
                            self.param[j] = 10. ** eval_map[j]
                    elif all(k in self.param for k in ['pNH3', 'dNH3', 'crNH3']):
                        self.param['Pa_top'] = 10. ** self.param['pNH3']
                        self.param['clda_depth'] = 10. ** self.param['dNH3']
                        self.param['CR_NH3'] = 10. ** self.param['crNH3']

                if gps in ('volume_mixing_ratio', 'vmr') and (self.param.get('gas_fill') is not None):
                    # Ensure filler completes the sum to unity
                    s = 0.0
                    for mol in self.param['fit_molecules']:
                        s += 10. ** eval_map[mol]
                        self.param['vmr_' + mol] = 10. ** eval_map[mol]
                    self.param['vmr_' + self.param['gas_fill']] = max(0.0, 1.0 - s)
                elif gps in ('centered_log_ratio', 'clr'):
                    self.param, _ = clr_to_vmr(self.param, eval_map)
                elif gps == 'partial_pressure':
                    pp = []
                    for mol in self.param['fit_molecules']:
                        pp.append(10.0 ** eval_map[mol])
                    self.param['P0'] = np.sum(pp)
                    for mol in self.param['fit_molecules']:
                        self.param['vmr_' + mol] = (10.0 ** eval_map[mol]) / self.param['P0']

                if self.param['fit_T'] and self.param['PT_profile_type'] == 'parametric':
                    if 'kappa_th' in eval_map:
                        self.param['kappa_th'] = 10. ** eval_map['kappa_th']
                    if 'gamma' in eval_map:
                        self.param['gamma'] = 10. ** eval_map['gamma']
                    if 'beta' in eval_map:
                        self.param['beta'] = eval_map['beta'] + 0.0
                    if self.param.get('fit_Tint') and 'Tint' in eval_map:
                        self.param['Tint'] = eval_map['Tint'] + 0.0

                if 'Rp' in eval_map.keys():
                    self.param['Rp'] = eval_map['Rp'] + 0.0
                if 'Mp' in eval_map.keys():
                    self.param['Mp'] = eval_map['Mp'] + 0.0
                if 'gp' in eval_map.keys():
                    if self.param['fit_g']:
                        self.param['gp'] = 10. ** (eval_map['gp'] - 2.0)
                    else:
                        self.param['gp'] = eval_map['gp'] + 0.0
                else:
                    self.param['gp'] = None

                if self.param['gp'] is not None and self.param['Mp'] is not None and self.param['Rp'] is None:
                    self.param['Rp'] = (np.sqrt((const.G.value * const.M_jup.value * self.param['Mp']) / self.param['gp'])) / const.R_jup.value
                elif self.param['gp'] is not None and self.param['Rp'] is not None and self.param['Mp'] is None:
                    self.param['Mp'] = ((self.param['gp'] * ((self.param['Rp'] * const.R_jup.value) ** 2.)) / const.G.value) / const.M_jup.value
                elif self.param['Mp'] is not None and self.param['Rp'] is not None and self.param['gp'] is None:
                    self.param['gp'] = (const.G.value * const.M_jup.value * self.param['Mp']) / ((const.R_jup.value * self.param['Rp']) ** 2.)
                elif self.param['Rp'] is not None and self.param['Mp'] is not None and self.param['gp'] is not None:
                    pass

                for j in other_params:
                    if j in eval_map.keys():
                        self.param[j] = eval_map[j] + 0.0
                if 'cld_frac' in eval_map:
                    if self.param['fit_cld_frac']:
                        self.param['cld_frac'] = eval_map['cld_frac'] + 0.0
                    else:
                        self.param['cld_frac'] = eval_map['cld_frac'] + 0.0
                    if self.param['cld_frac'] > 1.0 or self.param['cld_frac'] < 0.0:
                        raise ValueError("The sampled cloud fraction must be between [0.0, 1.0]. Please check 'cld_frac_range'.")
                if 'snr' in eval_map:
                    self.param['snr'] = eval_map['snr'] + 0.0
                    if self.param['snr'] <= 0.0:
                        raise ValueError("The sampled SNR must be positive. Please check 'snr_range'.")

                row_is_gaseous = False
                if self.param.get('mixed_planet_class', False):
                    row_is_gaseous = bool(self.param['Mp'] >= PLANET_CLASS_MASS_THRESHOLD_MJ)
                    self.param['is_gaseous'] = row_is_gaseous
                    self.param['gas_par_space'] = 'vmr'
                    self.param['gas_fill'] = 'H2'
                    self.param['fit_H2'] = False
                    if row_is_gaseous:
                        self.param['rocky'] = False
                        self.param['gaseous_planet'] = True
                        self.param['fit_p0'] = False
                        self.param['P0'] = 10. ** 11.5
                        self.param['fit_ag'] = False
                        self.param['Ag'] = 0.0
                        if self.param['surface_albedo_parameters'] == int(3):
                            self.param['Ag1'] = 0.0
                            self.param['Ag2'] = 0.0
                        elif self.param['surface_albedo_parameters'] == int(5):
                            self.param['Ag1'] = 0.0
                            self.param['Ag2'] = 0.0
                            self.param['Ag3'] = 0.0
                    else:
                        self.param['rocky'] = True
                        self.param['gaseous_planet'] = False
                        self.param['fit_p0'] = base_class_values['fit_p0']
                        self.param['fit_ag'] = base_class_values['fit_ag']
                        if 'P0' not in eval_map and base_class_values['P0'] is not None:
                            self.param['P0'] = base_class_values['P0']
                        for surf_key in ['Ag', 'Ag1', 'Ag2', 'Ag3', 'Ag_x1', 'Ag_x2']:
                            if surf_key not in eval_map and base_class_values[surf_key] is not None:
                                self.param[surf_key] = base_class_values[surf_key]

                Ls = (self.param['Rs'] ** 2.) * ((self.param['Ts'] / 5760.) ** 4.)
                F_ave = Ls / (self.param['major-a'] ** 2.)
                a_ave = 1. / (F_ave ** 0.5)
                self.param['equivalent_a'] = a_ave
                self.param['Tirr'] = 394.109 / (a_ave ** 0.5)

                if 'Tp' not in eval_map.keys():
                    if copy_tp is not None:
                        self.param['Tp'] = copy_tp + 0.0
                    else:
                        t1 = ((self.param['Rs'] * const.R_sun.value) / (2. * self.param['major-a'] * const.au.value)) ** 0.5
                        self.param['Tp'] = t1 * ((1 - 0.3) ** 0.25) * self.param['Ts']

                if "phi" not in eval_map.keys():
                    self.param['phi'] = math.pi * copy_phi / 180.0
                else:
                    self.param['phi'] = math.pi * self.param['phi'] / 180.0

                # Generate spectrum for this sample on this rank
                wl, model = forward(
                    self.param,
                    evaluation=None,
                    retrieval_mode=False,
                    core_number=rank,
                    albedo_calc=self.param['albedo_calc'],
                    fp_over_fs=self.param['fp_over_fs'],
                    canc_metadata=True
                )

                if self.param['cld_frac'] != 1.0 and (self.param['fit_wtr_cld'] or self.param['fit_amm_cld']):
                    fit_wtr_cld = self.param['fit_wtr_cld']
                    fit_amm_cld = self.param['fit_amm_cld']
                    self.param['fit_wtr_cld'] = False
                    self.param['fit_amm_cld'] = False
                    self.param['ret_mode'] = True
                    model_no_cld = forward(self.param, retrieval_mode=self.param['ret_mode'], albedo_calc=self.param['albedo_calc'], fp_over_fs=self.param['fp_over_fs'], canc_metadata=self.param['canc_metadata'])
                    self.param['fit_wtr_cld'] = fit_wtr_cld
                    self.param['fit_amm_cld'] = fit_amm_cld
                    self.param['ret_mode'] = False
                    model = (self.param['cld_frac'] * model) + ((1.0 - self.param['cld_frac']) * model_no_cld)

                data = np.array([wl, model]).T
                errorbars = None
                if self.param['add_noise'] and not self.param['albedo_calc']:
                    if self.param.get('noise_model') != 0:
                        raise ValueError("GEN_DATASET currently supports add_noise only with noise_model=0.")
                    self.param.pop('alpha1', None)
                    noisy_data = add_noise(self.param, data, noise_model=0)
                    data = noisy_data
                    model = noisy_data[:, 1]
                    errorbars = noisy_data[:, 2]

                # Save spectrum and sampled parameters to a JSON file for this sample
                # Build parameters payload: prefix gas keys with 'vmr_'
                payload_params = {}

                if gps != 'partial_pressure':
                    gas_fill = self.param.get('gas_fill')
                    payload_params['P0'] = self.param['P0'] + 0.0
                    if gas_fill is not None:
                        payload_params['vmr_' + str(gas_fill)] = self.param['vmr_' + str(gas_fill)]
                        payload_params[str(gas_fill) + '_filler'] = self.param['vmr_' + str(gas_fill)]
                else:
                    gas_fill = None

                for k, v in eval_map.items():
                    if (k in self.param['fit_molecules']) or (gas_fill is not None and k == gas_fill):
                        payload_params['vmr_' + str(k)] = self.param['vmr_' + str(k)]
                    else:
                        # Keep other sampled parameters as-is (e.g., 'Rp', 'P0', 'ag1', ...)
                        payload_params[str(k)] = self.param[str(k)]
                if self.param['add_noise']:
                    payload_params['snr'] = self.param['snr'] + 0.0

                record = {
                    'index': int((start_index if 'start_index' in locals() else 0) + i),
                    'wavelength': self.param['wave_file'],
                    'data': {
                        'spectrum': np.asarray(model, dtype=float).tolist(),
                        'errorbars': (
                            np.asarray(errorbars, dtype=float).tolist()
                            if errorbars is not None
                            else None
                        ),
                    },
                    'parameters': payload_params,
                }
                if self.param.get('mixed_planet_class', False):
                    record['is_gaseous'] = bool(row_is_gaseous)
                    record['planet_class'] = 'gaseous' if row_is_gaseous else 'rocky'

                # One file per sample to avoid MPI contention
                gidx = int((start_index if 'start_index' in locals() else 0) + i)
                fname = os.path.join(self.param['out_dir'], f'sample_{gidx:07d}.json')
                with open(fname, 'w') as f:
                    json.dump(record, f, separators=(',', ':'), ensure_ascii=False)

            # Guard against improper exit
            if MPIimport:
                MPI.COMM_WORLD.Barrier()
