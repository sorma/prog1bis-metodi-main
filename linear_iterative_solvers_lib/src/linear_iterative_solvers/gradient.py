from __future__ import annotations

import time

import numpy as np
import scipy.sparse

from linear_iterative_solvers.base import BaseSolver, SolverResult


class GradientSolver(BaseSolver):

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

        # Calcolo della norma del termine noto per il criterio di arresto
        b_norm = float(np.linalg.norm(b))
        
        # Gestione del caso con b nullo
        if b_norm < np.finfo(float).eps:
            return SolverResult(x=x, iterations=0, converged=True, residual_history=[0.0], elapsed_time=0.0)

        # Inizializzazione del residuo r0 = b - A*x0. Poiché x0=0, r0 coincide con b.
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

        # Ciclo iterativo del metodo del Gradiente
        for k in range(1, max_iter + 1):
            # Calcolo del prodotto matrice-vettore
            Ar = A @ r
            
            # Calcolo dei prodotti scalari necessari per determinare il passo ottimale alpha
            r_dot_r = float(r @ r)
            r_dot_Ar = float(r @ Ar)

            # Calcolo della lunghezza del passo alpha che minimizza la funzione quadratica lungo la direzione r
            alpha = r_dot_r / r_dot_Ar
            
            # Aggiornamento della soluzione
            x = x + alpha * r

            # Aggiornamento del residuo
            if k % 50 == 0:
                r = b - A @ x
            else:
                r = r - alpha * Ar

            # Calcolo della norma del residuo relativo per il criterio di convergenza
            res = float(np.linalg.norm(r) / b_norm)
            residual_history.append(res)
            
            # Se l'errore è inferiore alla tolleranza richiesta, interrompiamo il ciclo
            if res < tol:
                converged = True
                break

        # Calcolo quanto tempo è passato
        elapsed = time.perf_counter() - t_start

        return SolverResult(
            x=x,
            iterations=k,
            converged=converged,
            residual_history=residual_history,
            elapsed_time=elapsed,
        )