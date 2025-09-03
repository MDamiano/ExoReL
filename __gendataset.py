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
    print('MPI disabled')


class GEN_DATASET:
    def __init__(self, par):
        self.param = copy.deepcopy(par)
        self.param = pre_load_variables(self.param)

    def run(self):
        if MPIimport and MPIrank == 0:
            print(f"Running ExoReL – version {__version__}")
            npar, par = detect_gen_npar(self.param)
        
            # Generate unit-hypercube samples X in [0,1)^npar
            if self.param['optimizer'] == 'sobol':
                sampler = sp.stats.qmc.Sobol(d=npar, scramble=True)
                # Best practice: draw 2**m points
                X = sampler.random_base2(m=int(np.ceil(np.log2(self.param['n_spectra']))))
            
            else:
                X = np.random.random((self.param['n_spectra'], npar))

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
            lines = [f"[{i:02d}] {name}: {lo} -> {hi}" for i, (name, lo, hi) in enumerate(zip(par, lower, upper))]
            print(f"Synthesizing {X.shape[0]} spectra using {npar} sampled parameters.")
            print("Columns mapping (index, parameter, lower, upper):\n" + "\n".join(lines))
            X_scaled = sp.stats.qmc.scale(X, lower, upper)

        # Build the design matrix header (rank 0 only). Actual CSV write happens later.
        if MPIimport and MPIrank == 0:
            header = 'index,' + ','.join([str(p) for p in par])

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
                expected_cols = ['index'] + [str(p) for p in par]
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
            data = np.concatenate((idx, X_scaled), axis=1)

            if append_mode:
                with open(csv_path, 'ab') as f:
                    np.savetxt(f, data, delimiter=',', comments='', fmt='%g')
            else:
                header = 'index,' + ','.join([str(p) for p in par])
                np.savetxt(csv_path, data, delimiter=',', header=header, comments='', fmt='%g')

            # Update sidecar metadata
            meta = {
                'columns': ['index'] + list(par),
                'n_rows': int(start_index + nrows),
                'n_cols': int(data.shape[1] - 1),
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

        # Determine work partition for this rank
        total = X_scaled.shape[0]
        size = MPIsize if MPIimport else 1
        rank = MPIrank if MPIimport else 0
        start = (total * rank) // size
        end = (total * (rank + 1)) // size

        gps = self.param.get('gas_par_space')

        other_params = ["Rs", "Ms", "Ts", "major-a", "Tp", "cld_frac", "Ag", "Ag1", "Ag2", "Ag3", "Ag_x1", "Ag_x2", "phi", "p_size"]

        # Tight loop over this rank's chunk
        for i in range(start, end):
            row = X_scaled[i]

            # Build evaluation dict mapping each parameter to its sampled value
            eval_map = {}
            for j, name in enumerate(par):
                if name.endswith('_range'):
                    key = name[:-6]
                else:
                    # Molecule names and other direct keys
                    key = name
                eval_map[key] = row[j]
            
            if self.param['fit_wtr_cld']:
                for j in ['Pw_top', 'cldw_depth', 'CR_H2O']:
                        self.param[j] = 10. ** eval_map[j]
            if self.param['fit_amm_cld']:
                for j in ['Pa_top', 'clda_depth', 'CR_NH3']:
                        self.param[j] = 10. ** eval_map[j]

            if gps in ('volume_mixing_ratio', 'vmr') and (self.param.get('gas_fill') is not None):
                # Ensure filler completes the sum to unity
                s = 0.0
                for mol in self.param['fit_molecules']:
                    s += 10. ** eval_map[mol]
                    self.param['vmr_' + mol] = 10. ** eval_map[mol]
                self.param['vmr_' + self.param['gas_fill']] = max(0.0, 1.0 - s)
            elif gps == 'partial_pressure':
                pp = []
                for mol in self.param['fit_molecules']:
                    pp.append(10.0 ** eval_map[mol])
                self.param['P0'] = np.sum(pp)
                for mol in self.param['fit_molecules']:
                    self.param['vmr_' + mol] = (10.0 ** eval_map[mol]) / self.param['P0']

            if 'Rp' in eval_map.keys():
                self.param['Rp'] = eval_map['Rp'] + 0.0
            if 'Mp' in eval_map.keys():
                self.param['Mp'] = eval_map['Mp'] + 0.0
            if 'gp' in eval_map.keys():
                self.param['gp'] = eval_map['gp'] + 0.0

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

            self.param = par_and_calc(self.param)

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

            if self.param['cld_frac'] != 1.0 and self.param['fit_wtr_cld']:
                self.param['fit_wtr_cld'] = False
                self.param['ret_mode'] = True
                model_no_cld = forward(self.param, retrieval_mode=self.param['ret_mode'], albedo_calc=self.param['albedo_calc'], fp_over_fs=self.param['fp_over_fs'], canc_metadata=self.param['canc_metadata'])
                self.param['fit_wtr_cld'] = True
                self.param['ret_mode'] = False
                model = (self.param['cld_frac'] * model) + ((1.0 - self.param['cld_frac']) * model_no_cld)

            # Save spectrum and sampled parameters to a JSON file for this sample
            # Build parameters payload: prefix gas keys with 'vmr_'
            payload_params = {}

            if gps != 'partial_pressure':
                gas_fill = self.param.get('gas_fill')
                payload_params['P0'] = self.param['P0'] + 0.0
            else:
                gas_fill = None

            for k, v in eval_map.items():
                if (k in self.param['fit_molecules']) or (gas_fill is not None and k == gas_fill):
                    payload_params['vmr_' + str(k)] = self.param['vmr_' + str(k)]
                else:
                    # Keep other sampled parameters as-is (e.g., 'Rp', 'P0', 'ag1', ...)
                    payload_params[str(k)] = self.param[str(k)]

            record = {
                'index': int((start_index if 'start_index' in locals() else 0) + i),
                # 'wavelength': np.asarray(wl, dtype=float).tolist(),
                'spectrum': np.asarray(model, dtype=float).tolist(),
                'parameters': payload_params,
            }

            # One file per sample to avoid MPI contention
            gidx = int((start_index if 'start_index' in locals() else 0) + i)
            fname = os.path.join(self.param['out_dir'], f'sample_{gidx:07d}.json')
            with open(fname, 'w') as f:
                json.dump(record, f, separators=(',', ':'), ensure_ascii=False)

        # Guard against improper exit
        if MPIimport:
            MPI.COMM_WORLD.Barrier()
