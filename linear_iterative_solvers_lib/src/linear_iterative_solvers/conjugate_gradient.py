from __future__ import annotations

import time

import numpy as np
import scipy.sparse

from linear_iterative_solvers.base import BaseSolver, SolverResult


class ConjugateGradientSolver(BaseSolver):
    
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

        # Inizializzazione del residuo r0 e della direzione p0
        r = b.copy()
        p = r.copy()
        
        # Prodotto scalare del residuo per il calcolo di alpha e beta
        r_dot_r = float(r @ r)

        # Calcolo dell'errore relativo iniziale
        res = float(np.linalg.norm(r) / b_norm)
        residual_history.append(res)
        
        # Controllo se l'errore relativo iniziale è già entro la tolleranza
        if res < tol:
            return SolverResult(x=x, iterations=0, converged=True, residual_history=residual_history, elapsed_time=0.0)

        t_start = time.perf_counter()
        converged = False
        k = 0

        # Ciclo iterativo del metodo del Gradiente Coniugato
        for k in range(1, max_iter + 1):
            # Calcolo del prodotto matrice-vettore
            Ap = A @ p
            
            # Denominatore del passo alpha
            p_dot_Ap = float(p @ Ap)

            # Calcolo della lunghezza del passo alpha
            alpha = r_dot_r / p_dot_Ap

            # Aggiornamento della soluzione e del residuo
            x = x + alpha * p
            
            # Aggiornamento del residuo: ogni 50 iterazioni ricalcoliamo il residuo reale
            # per evitare l'accumulo di errori di arrotondamento in virgola mobile.
            if k % 50 == 0:
                r_new = b - A @ x
            else:
                r_new = r - alpha * Ap

            # Calcolo del coefficiente beta per la nuova direzione coniugata
            r_dot_r_new = float(r_new @ r_new)
            beta = r_dot_r_new / r_dot_r

            # Costruzione della nuova direzione di ricerca e aggiornamento variabili
            p = r_new + beta * p
            r = r_new
            r_dot_r = r_dot_r_new

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