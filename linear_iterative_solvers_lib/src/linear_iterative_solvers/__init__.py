from linear_iterative_solvers._version import __version__
from linear_iterative_solvers.base import BaseSolver, SolverResult
from linear_iterative_solvers.jacobi import JacobiSolver
from linear_iterative_solvers.gauss_seidel import GaussSeidelSolver
from linear_iterative_solvers.gradient import GradientSolver
from linear_iterative_solvers.conjugate_gradient import ConjugateGradientSolver

__all__ = [
    "__version__",
    # Base
    "BaseSolver",
    "SolverResult",
    # Solvers
    "JacobiSolver",
    "GaussSeidelSolver",
    "GradientSolver",
    "ConjugateGradientSolver",
]
