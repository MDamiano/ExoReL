from .__basics import *
from .__utils import *
from .__forward import *

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
    print('SOME ERRORS OCCURRED - MultiNest library is not loaded.')
    raise ImportError


class GEN_DATASET:
    def __init__(self, par):
        self.param = copy.deepcopy(par)
        self.param = pre_load_variables(self.param)

    def run(self):
        npar, par = detect_gen_npar(self.param)
        
        if self.param['optimizer'] == 'sobol':
            sampler = sp.stats.qmc.Sobol(d=npar, scramble=True)

            # Best practice: draw 2**m points
            X = sampler.random_base2(m=int(np.ceil(np.log2(self.param['n_spectra']))))     # 2**10 = 1024 points in [0,1)^d
            print(X.shape)
            sys.exit()

            # # scale to parameter bounds
            # lower = [0, -3, 10, 0.1, 1.0]
            # upper = [1,  3, 50, 1.0, 2.0]
            # X_scaled = qmc.scale(X, lower, upper)
