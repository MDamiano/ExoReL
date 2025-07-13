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
        pass
