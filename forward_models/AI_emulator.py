from ExoReL.__basics import *
from ExoReL.__utils import *
from pathlib import Path

import torch

from .network_routines.model import AlbedoTransformer, ModelConfig


class FORWARD_AI:
    """Neural-network based forward model using pretrained ExoReL transformers."""

    _LOG10_KEYS = {"Pw_top", "cldw_depth", "CR_H2O", "Pa_top", "clda_depth", "CR_NH3", "p_size"}
    _VALUE_EPS = 1e-20
    _MODEL_CACHE = {}
    _STATS_CACHE = {}
    _STAT_FILES = ('inference_stats.json', 'network_config.json', 'network_stats.json')

    def __init__(self, par):
        self.param = copy.deepcopy(par)
        try:
            self.ai_dir = Path(self.param['ai_dir']).expanduser()
        except KeyError as exc:
            raise KeyError('param must contain "ai_dir" to run the AI model.') from exc
        if not self.ai_dir.exists():
            raise FileNotFoundError('AI directory not found: ' + str(self.ai_dir))

        self.model_path = self._select_model_path()
        self.param_columns, self.param_min, self.param_scale = self._load_inference_stats()
        self.param_min = self.param_min.astype(np.float32, copy=False)
        self.param_scale = self.param_scale.astype(np.float32, copy=False)
        self.device = torch.device('cpu')
        self.model = self._build_model(len(self.param_columns))

    def _select_model_path(self):
        filename = 'rocky_net.pt' if bool(self.param.get('rocky')) else 'gas_net.pt'
        path = self.ai_dir / filename
        if not path.is_file():
            raise FileNotFoundError('Model checkpoint not found: ' + str(path))
        return path

    def _load_inference_stats(self):
        stats_path = None
        for name in self._STAT_FILES:
            candidate = self.ai_dir / name
            if candidate.is_file():
                stats_path = candidate
                break
        if stats_path is None:
            raise FileNotFoundError(
                'Inference statistics file not found in ' + str(self.ai_dir) +
                '. Expected one of: ' + ', '.join(self._STAT_FILES)
            )

        key = str(stats_path.resolve())
        cached = self._STATS_CACHE.get(key)
        if cached is None:
            with stats_path.open('r', encoding='utf-8') as fh:
                payload = json.load(fh)
            if 'param_columns' in payload:
                stats_payload = payload
            else:
                stats_payload = payload.get('inference_stats')
            if stats_payload is None:
                raise ValueError('No inference statistics found in ' + str(stats_path))
            columns = stats_payload.get('param_columns')
            minimum = stats_payload.get('minimum')
            scale = stats_payload.get('scale')
            if columns is None or minimum is None or scale is None:
                raise ValueError('Inference stats file missing required fields: ' + str(stats_path))
            columns = list(columns)
            minimum = np.asarray(minimum, dtype=np.float32)
            scale = np.asarray(scale, dtype=np.float32)
            if minimum.shape[0] != len(columns) or scale.shape[0] != len(columns):
                raise ValueError('Mismatch between stats vector lengths and columns in ' + str(stats_path))
            if np.any(scale <= 0.0):
                raise ValueError('Non-positive scale values in inference stats: ' + str(stats_path))
            minimum = minimum.astype(np.float32, copy=False)
            scale = scale.astype(np.float32, copy=False)
            cached = (tuple(columns), minimum.copy(), scale.copy())
            self._STATS_CACHE[key] = cached
        columns, minimum, scale = cached
        return list(columns), minimum.copy(), scale.copy()

    def _build_model(self, param_dim):
        state_dict = self._load_state_dict(param_dim)
        model = AlbedoTransformer(param_dim=param_dim, config=ModelConfig())
        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        model.eval()
        return model

    def _load_state_dict(self, param_dim):
        key = (str(self.model_path.resolve()), int(param_dim))
        cached = self._MODEL_CACHE.get(key)
        if cached is None:
            try:
                payload = torch.load(self.model_path, map_location='cpu', weights_only=False)
            except TypeError:
                payload = torch.load(self.model_path, map_location='cpu')
            if isinstance(payload, dict) and 'state_dict' in payload:
                payload = payload['state_dict']
            if not isinstance(payload, dict):
                raise ValueError('Unexpected checkpoint format in ' + str(self.model_path))
            cached = {k: v.clone().cpu() for k, v in payload.items()}
            self._MODEL_CACHE[key] = cached
        return {k: v.clone() for k, v in cached.items()}

    def _wavelength_grid(self):
        if bool(self.param.get('wl_native')):
            wl_grid = np.asarray(self.param.get('wl_C_grid'), dtype=float)
            if wl_grid.size == 0:
                raise ValueError('wl_C_grid is empty; cannot build native wavelength grid.')
            start = int(self.param.get('start_c_wl_grid', 0))
            stop = int(self.param.get('stop_c_wl_grid', wl_grid.size - 1))
            start = max(start, 0)
            stop = min(stop, wl_grid.size - 1)
            if stop < start:
                start, stop = 0, wl_grid.size - 1
            return wl_grid[start: stop + 1].astype(np.float32)

        spectrum = self.param.get('spectrum')
        if spectrum is None:
            raise KeyError('param must include a "spectrum" entry to derive the wavelength grid.')
        wl = spectrum.get('wl')
        if wl is None and self.param.get('obs_numb') is not None:
            obs_key = str(self.param['obs_numb'])
            obs_spec = spectrum.get(obs_key)
            if obs_spec is not None:
                wl = obs_spec.get('wl')
        if wl is None:
            raise KeyError('Unable to locate wavelength grid in param["spectrum"].')
        return np.asarray(wl, dtype=np.float32)

    def _extract_scalar(self, key):
        if key == 'phi':
            value = self.param.get('phi')
            if value is None:
                raise KeyError('Parameter "phi" missing in param for AI inference.')
            return float(math.degrees(value))

        value = self.param.get(key)
        if value is None:
            raise KeyError('Parameter "' + str(key) + '" missing in param for AI inference.')
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                raise ValueError('Parameter "' + str(key) + '" is empty.')
            value = value[-1]
        if isinstance(value, np.ndarray):
            if value.size == 0:
                raise ValueError('Parameter "' + str(key) + '" is empty.')
            value = value.reshape(-1)[-1]
        return float(value)

    def _encode_molecule(self, mol, gps):
        key = 'vmr_' + mol
        value = self.param.get(key)
        if value is None:
            if mol == self.param.get('gas_fill'):
                value = self.param.get(key, 0.0)
            else:
                raise KeyError('Mixing ratio for molecule "' + mol + '" missing in param.')
        if isinstance(value, (list, tuple)):
            value = value[-1]
        if isinstance(value, np.ndarray):
            value = value.reshape(-1)[-1]
        value = float(value)
        value = max(value, self._VALUE_EPS)
        if gps in ('volume_mixing_ratio', 'vmr'):
            return float(np.log10(value))
        if gps == 'partial_pressure':
            P0 = self.param.get('P0')
            if P0 is None:
                raise KeyError('P0 is required for partial_pressure gas_par_space.')
            return float(np.log10(max(value * float(P0), self._VALUE_EPS)))
        return float(value)

    def _build_param_vector(self):
        gps = self.param.get('gas_par_space')
        xtgt = []
        for cname in self.param_columns:
            if cname.endswith('_range'):
                key = cname[:-6]
                value = self._extract_scalar(key)
                if key in self._LOG10_KEYS:
                    value = np.log10(max(value, self._VALUE_EPS))
                xtgt.append(float(value))
            else:
                xtgt.append(self._encode_molecule(cname, gps))
        return np.asarray(xtgt, dtype=np.float32)

    def run_forward(self):
        lam = self._wavelength_grid()
        params_vec = self._build_param_vector()
        params_norm = (params_vec - self.param_min) / self.param_scale
        params_tensor = torch.from_numpy(params_norm.astype(np.float32)).unsqueeze(0).to(self.device)
        lam_tensor = torch.from_numpy(lam).unsqueeze(0).to(self.device)
        mask = torch.ones((1, lam_tensor.shape[1]), dtype=torch.bool, device=self.device)

        with torch.no_grad():
            self.model.fourier.fit(lam_tensor.squeeze(0).cpu())
            prediction = self.model(params_tensor, lam_tensor, mask=mask)

        albedo = prediction.squeeze(0).cpu().numpy().astype(float)
        return lam.astype(float), albedo