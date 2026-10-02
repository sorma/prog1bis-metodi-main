from __future__ import annotations

import time

import numpy as np
import scipy.sparse

from linear_iterative_solvers.base import BaseSolver, SolverResult


class JacobiSolver(BaseSolver):

    def solve(
        self,
        A: scipy.sparse.spmatrix,
        b: np.ndarray,
        tol: float,
        max_iter: int,
    ) -> SolverResult:
        # Controllo che la matrice e il termine noto siano validi
        self.validate(A, b, mat_iter=max_iter)

        # Conversione del termine noto in array NumPy
        b = np.asarray(b, dtype=np.float64)
        # Dimensione del sistema lineare
        n = b.shape[0]
        
        # Punto di partenza x0 inizializzato a zero
        x = np.zeros(n, dtype=np.float64)
        residual_history: list[float] = []

        # Estrazione della diagonale D e della matrice di scarto R (A = D + R)
        d = A.diagonal()
        R = A - scipy.sparse.diags(d)

        # Calcolo della norma del termine noto per il criterio di arresto
        b_norm = float(np.linalg.norm(b))
        
        # Gestione del caso con b nullo
        if b_norm < np.finfo(float).eps:
            return SolverResult(x=x, iterations=0, converged=True, residual_history=[0.0], elapsed_time=0.0)

        # Inizializzazione del residuo r0
        r = b.copy()
        
        # Calcolo dell'errore relativo iniziale
        res = float(np.linalg.norm(r) / b_norm)
        residual_history.append(res)
        
        # Controllo se l'errore relativo iniziale è già entro la tolleranza
        if res < tol:
            return SolverResult(x=x, iterations=0, converged=True, residual_history=residual_history, elapsed_time=0.0)


        t_start = time.perf_counter()
        converged = False
        k = 0

        # Ciclo iterativo del metodo di Jacobi
        for k in range(1, max_iter + 1):
            # Aggiornamento della soluzione
            x = x + r / d

            # Calcolo del prodotto matrice-vettore
            Rx = R @ x
            
            # Aggiornamento del residuo
            r = b - Rx - d * x
            
            # Calcolo della norma del residuo relativo per il criterio di convergenza
            res = float(np.linalg.norm(r) / b_norm)
            residual_history.append(res)

            # Se l'errore è inferiore alla tolleranza richiesta, interrompiamo il ciclo
            if res < tol:
                converged = True
                break

        elapsed = time.perf_counter() - t_start

        return SolverResult(
            x=x,
            iterations=k,
            converged=converged,
            residual_history=residual_history,
            elapsed_time=elapsed,
        )
