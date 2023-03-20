import scipy.sparse.linalg as spl


try:
    from pyMKL import pardisoSolver
    PYMKL_ENABLED = True
except ImportError:
    PYMKL_ENABLED = False


def linear_solve(A, b, method='direct'):
    if method == 'direct':
        if PYMKL_ENABLED:
            return direct_solver_mkl(A, b)
        else:
            return direct_solver_spl(A, b)
    else:
        raise NotImplementedError("Actually only MKL or Scipy Sparse solver"
                                  + " are available.")


def direct_solver_mkl(A, b):
    pSolve = pardisoSolver(A.tocsr(), mtype=13)
    pSolve.factor()
    x = pSolve.solve(b)
    pSolve.clear()
    return x


def direct_solver_spl(A, b):
    x = spl.spsolve(A, b)
    return x
