import numpy as np
import scipy.sparse

def run_solver(solver, A, b, tol: float, max_iter: int) -> dict:
    # Esegue un solver su A, b e restituisce le metriche.
    result = solver.solve(A, b, tol=tol, max_iter=max_iter)

    return {
        "x": result.x,
        "iterations": result.iterations,
        "converged": result.converged,
        "elapsed_time": result.elapsed_time,
    }

def make_b( A: scipy.sparse.spmatrix) -> tuple[np.ndarray, np.ndarray]:
    # Costruisce la soluzione esatta e il vettore termine noto.
    n = A.shape[0]
    x_exact = np.ones(n, dtype=np.float64)
    b: np.ndarray = np.asarray(A @ x_exact, dtype=np.float64).ravel()
    return b, x_exact

def relative_error(x_approx: np.ndarray, x_exact: np.ndarray) -> float:
    # Calcola l'errore relativo tra la soluzione approssimata e quella esatta.
    norm_exact = float(np.linalg.norm(x_exact))
    if norm_exact == 0.0:
        return float(np.linalg.norm(x_exact - x_approx))
    return float(np.linalg.norm(x_exact - x_approx) / norm_exact)