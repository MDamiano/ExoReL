from .__basics import *
from .__utils import *
from .__forward import *
from . import __version__
from .network_routines.train import run_training

path = os.path.abspath(__file__)
pkg_dir = os.path.dirname(path) + '/'


class RETRIEVAL:
    def __init__(self):
        param = default_parameters()
        param['pkg_dir'] = pkg_dir
        self.param = copy.deepcopy(param)
        self.param['ret_mode'] = True

    def run_retrieval(self, parfile):
        self.param = read_parfile(self.param, parfile, json_format=True)
        self.param = setup_param_dict(self.param)
        if self.param['optimizer'] == 'multinest':
            from ExoReL.__multinest import MULTINEST # type: ignore
            bayes = MULTINEST(self.param)
            bayes.run_retrieval()
        else:
            pass


class CREATE_SPECTRUM:
    def __init__(self, canc_metadata=False, verbose=True):
        param = default_parameters()
        param['pkg_dir'] = pkg_dir
        self.param = copy.deepcopy(param)
        self.param['ret_mode'] = False
        self.canc_metadata = canc_metadata
        self.param['verbose'] = verbose

    def run_forward(self, parfile):
        if self.param['verbose']:
            print(f"Running ExoReL – version {__version__}")
        self.param = read_parfile(self.param, parfile)
        self.param = setup_param_dict(self.param)
        self.param = par_and_calc(self.param)
        self.param = load_input_spectrum(self.param)

        if not self.param['albedo_calc'] and not self.param['fp_over_fs']:
            self.param = take_star_spectrum(self.param)
        self.param = pre_load_variables(self.param)

        if self.param['verbose']:
            print('Calculating the Planetary Albedo')

            print('Parameters:')

        try:
            self.param['P0'] = self.param['P0']
        except KeyError:
            print('The surface pressure level (P0) should be expressed')
            sys.exit()

        if self.param['fit_wtr_cld']:
            self.param['Pw_top'] = 10. ** self.param['pH2O']
            self.param['cldw_depth'] = 10. ** self.param['dH2O']
            self.param['CR_H2O'] = 10. ** self.param['crH2O']
        if self.param['fit_amm_cld']:
            self.param['Pa_top'] = 10. ** self.param['pNH3']
            self.param['clda_depth'] = 10. ** self.param['dNH3']
            self.param['CR_NH3'] = 10. ** self.param['crNH3']
        if self.param['fit_g']:
            self.param['gp'] = (10. ** (self.param['g'] - 2.0))
        if self.param['gp'] is not None and self.param['Mp'] is not None and self.param['Rp'] is None:
            self.param['Rp'] = (np.sqrt((const.G.value * const.M_jup.value * self.param['Mp']) / self.param['gp'])) / const.R_jup.value
        elif self.param['gp'] is not None and self.param['Rp'] is not None and self.param['Mp'] is None:
            self.param['Mp'] = ((self.param['gp'] * ((self.param['Rp'] * const.R_jup.value) ** 2.)) / const.G.value) / const.M_jup.value
        elif self.param['Mp'] is not None and self.param['Rp'] is not None and self.param['gp'] is None:
            self.param['gp'] = (const.G.value * const.M_jup.value * self.param['Mp']) / ((const.R_jup.value * self.param['Rp']) ** 2.)
        elif self.param['Rp'] is not None and self.param['Mp'] is not None and self.param['gp'] is not None:
            pass
        else:
            raise RuntimeError("Please define two among planetary radius, mass, or gravity. I cannot complete the calculation otherwise.")

        self.param['fit_molecules'] = []
        sumb = 0.0
        for mol in self.param['supported_molecules']:
            try:
                if self.param['vmr_' + mol] != 0.0:
                    self.param['fit_molecules'].append(mol)
                    sumb += self.param['vmr_' + mol]
            except KeyError:
                pass
        if self.param['gas_fill'] is not None:
            self.param['vmr_' + self.param['gas_fill']] = 1.0 - sumb
        else:
            pass

        if self.param['verbose']:
            if self.param['fit_wtr_cld']:
                print('Log(H2O_Ptop) \t = \t' + str(self.param['pH2O']))
                print('Log(H2O_D) \t = \t' + str(self.param['dH2O']))
                print('Log(H2O_CR) \t = \t' + str(self.param['crH2O']))
            if self.param['fit_amm_cld']:
                print('Log(NH3_Ptop) \t = \t' + str(self.param['pNH3']))
                print('Log(NH3_D) \t = \t' + str(self.param['dNH3']))
                print('Log(NH3_CR) \t = \t' + str(self.param['crNH3']))
            if self.param['fit_wtr_cld'] or self.param['fit_amm_cld']:
                print('Cloud fraction \t = \t' + str(self.param['cld_frac']))
            print('g \t\t = \t' + str(self.param['gp']))
            print('Mp \t\t = \t' + str(self.param['Mp']))
            print('Rp \t\t = \t' + str(self.param['Rp']))
            print('Tp \t\t = \t' + str(self.param['Tp']))
            if self.param['surface_albedo_parameters'] == int(1):
                print('Ag \t\t = \t' + str(self.param['Ag']))
            elif self.param['surface_albedo_parameters'] == int(3):
                print('Ag1 \t\t = \t' + str(self.param['Ag1']))
                print('Ag2 \t\t = \t' + str(self.param['Ag2']))
                print('Ag_x1 \t\t = \t' + str(self.param['Ag_x1']))
            elif self.param['surface_albedo_parameters'] == int(5):
                print('Ag1 \t\t = \t' + str(self.param['Ag1']))
                print('Ag2 \t\t = \t' + str(self.param['Ag2']))
                print('Ag3 \t\t = \t' + str(self.param['Ag3']))
                print('Ag_x1 \t\t = \t' + str(self.param['Ag_x1']))
                print('Ag_x2 \t\t = \t' + str(self.param['Ag_x2']))
            print('Log(P0) \t = \t' + str(np.log10(self.param['P0'])))
            print('phi \t\t = \t' + str(self.param['phi'] * 180.0 / math.pi))

            for mol in self.param['fit_molecules']:
                if mol == 'N2' and self.param['gas_fill'] == 'H2' and self.param['vmr_N2'] != 0:
                    print('VMR N2 \t\t = \t' + str(self.param['vmr_N2']))
                elif mol == 'N2' and self.param['gas_fill'] is None and self.param['vmr_N2'] != 0:
                    print('VMR N2 \t\t = \t' + str(self.param['vmr_N2']))
                elif mol == 'N2' and self.param['gas_fill'] == 'N2':
                    pass
                elif mol == 'O2' or mol == 'O3' or mol == 'CO':
                    print('VMR ' + mol + ' \t\t = \t' + str(self.param['vmr_' + mol]))
                else:
                    print('VMR ' + mol + ' \t = \t' + str(self.param['vmr_' + mol]))

            if self.param['gas_fill'] is not None and self.param['gas_fill'] == 'N2':
                print('VMR N2 \t\t = \t' + str(self.param['vmr_N2']))
            elif self.param['gas_fill'] is not None and self.param['gas_fill'] == 'H2':
                # print('VMR H2 \t\t = \t' + str(self.param['vmr_H2']))
                if self.param['rocky']:
                    print('VMR H2 \t\t = \t' + str(self.param['vmr_H2']))
                else:
                    if self.param['H2_He_ratio'] != 0:
                        print('VMR He \t\t = \t' + str(self.param['vmr_H2'] * (1.0 - self.param['H2_He_ratio'])))
                        print('VMR H2 \t\t = \t' + str(self.param['vmr_H2'] * self.param['H2_He_ratio']))
            else:
                pass

        time1 = time.time()

        wl, model = forward(self.param, retrieval_mode=self.param['ret_mode'], albedo_calc=self.param['albedo_calc'], fp_over_fs=self.param['fp_over_fs'], canc_metadata=self.canc_metadata)

        if self.param['cld_frac'] != 1.0 and self.param['fit_wtr_cld']:
            self.param['fit_wtr_cld'] = False
            self.param['ret_mode'] = True
            model_no_cld = forward(self.param, retrieval_mode=self.param['ret_mode'], albedo_calc=self.param['albedo_calc'], fp_over_fs=self.param['fp_over_fs'], canc_metadata=self.canc_metadata)
            self.param['fit_wtr_cld'] = True
            self.param['ret_mode'] = False
            model = (self.param['cld_frac'] * model) + ((1.0 - self.param['cld_frac']) * model_no_cld)

        time2 = time.time()
        if self.param['verbose']:
            elapsed((time2 - time1) * (10 ** 9))

        data = np.array([wl, model]).T

        if self.param['add_noise'] and not self.param['albedo_calc']:
            data = add_noise(self.param, data, noise_model=self.param['noise_model'])

        if self.param['spectrum']['bins'] and self.param['return_bins']:
            data = np.concatenate((np.array([self.param['spectrum']['wl_high']]).T, data), axis=1)
            data = np.concatenate((np.array([self.param['spectrum']['wl_low']]).T, data), axis=1)

        try:
            if not os.path.exists(self.param['out_dir']):
                os.mkdir(self.param['out_dir'])
            np.savetxt(self.param['out_dir'] + str(self.param['file_output_name']) + '.dat', data)
        except IOError or KeyError:
            if not os.path.exists(self.param['pkg_dir'] + 'Output/'):
                os.mkdir(self.param['pkg_dir'] + 'Output/')

            np.savetxt(self.param['pkg_dir'] + 'Output/spectrum.dat', data)


class CREATE_DATASET:
    def __init__(self, canc_metadata=True, verbose=False):
        param = default_parameters()
        param['pkg_dir'] = pkg_dir
        self.param = copy.deepcopy(param)
        self.param['ret_mode'] = False
        self.param['canc_metadata'] = canc_metadata
        self.param['verbose'] = verbose

    def run_forward(self, parfile):
        self.param = read_parfile(self.param, parfile, json_format=True)
        self.param = setup_param_dict(self.param)
        self.param = load_input_spectrum(self.param)
        from ExoReL.__gendataset import GEN_DATASET # type: ignore
        ob = GEN_DATASET(self.param)
        ob.run()


class TRAIN_NN:
    def __init__(self):
        param = default_parameters()
        param['pkg_dir'] = pkg_dir
        self.param = copy.deepcopy(param)

    def run_training(self, parfile):
        config = read_parfile(self.param, parfile, json_format=True)
        network_cfg = config.get('network_training')
        if not isinstance(network_cfg, dict):
            raise RuntimeError('Parfile must contain a "network_training" section with configuration values.')

        dataset_dir = network_cfg.get('dataset_dir') or config.get('dataset_dir')
        output_dir = (
            network_cfg.get('output_directory')
            or network_cfg.get('output_dir')
            or config.get('output_directory')
            or config.get('out_dir')
        )
        if dataset_dir is None or output_dir is None:
            raise RuntimeError('Training configuration must provide dataset_dir and output_directory.')

        cleaned_network_cfg = {
            key: value
            for key, value in network_cfg.items()
            if key not in {'dataset_dir', 'output_dir', 'output_directory'}
        }

        payload = {
            'dataset_dir': dataset_dir,
            'output_directory': output_dir,
            "out_net_name": config.get("out_net_name"),
            'network_training': cleaned_network_cfg,
        }
        run_training(payload)
