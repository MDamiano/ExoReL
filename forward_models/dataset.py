from ExoReL.__basics import *
from ExoReL.__utils import *
from pathlib import Path


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
                if key == 'phi':
                   val = self.param[key] * 180.0 / math.pi
                xtgt.append(float(val))
            else:
                # Molecule dimension
                mol = cname
                if gps in ('volume_mixing_ratio', 'vmr'):
                    v = self.param.get('vmr_' + mol)
                    try:
                        if len(v) > 0:
                            pass
                    except TypeError:
                        v = np.ones(len(self.param['P'])) * (10 ** (-11.5))  # Default tiny VMR if missing
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
        file_to_open = 1000

        if n_samples > file_to_open:
            # Compute squared distances to target and select k nearest
            d2 = np.sum((X - xtgt) ** 2, axis=1)
            nn_idx = np.argpartition(d2, file_to_open - 1)[:file_to_open]
            X_use = X[nn_idx]
            idx_use = idx[nn_idx]
        else:
            raise ValueError('Dataset too small for interpolation; need more than ' + str(file_to_open) + ' samples')

        Y, wave_file_id = self._load_spectra_matrix(idx_use)

        # 5) Check if inside convex hull and interpolate
        # Use Delaunay once and barycentric linear interpolation for speed
        # Build Delaunay on the working set and interpolate
        # If target lies outside this subset hull or triangulation fails, fall back to nearest
        y = None
        if X_use.shape[1] <= 8:  # Delaunay takes too much time in high-D
            tri = sp.spatial.Delaunay(X_use)
            simplex = tri.find_simplex(xtgt)
            if simplex >= 0:
                interp = sp.interpolate.LinearNDInterpolator(tri, Y)
                y = interp(xtgt)
        else:
            # Fallback to nearest interpolation on the working set
            y = sp.interpolate.NearestNDInterpolator(X_use, Y)(xtgt)

        alb_wl = self._load_wavelength_grid(wave_file_id)
        alb = np.asarray(y, dtype=float)

        return alb_wl, alb[0]