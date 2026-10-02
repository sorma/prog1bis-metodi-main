from __future__ import annotations

import time

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from linear_iterative_solvers.base import BaseSolver, SolverResult


class GaussSeidelSolver(BaseSolver):
    def __init__(self, use_scipy: bool = False):
        self.use_scipy = use_scipy

    def forward_substitution_csr(self, L: scipy.sparse.csr_matrix, b: np.ndarray) -> np.ndarray:
        n = b.shape[0]
        x = np.zeros(n, dtype=np.float64)
        
        # Estrazione dati dalla struttura CSR
        data = L.data
        indices = L.indices
        indptr = L.indptr
        
        # Risoluzione riga per riga
        for i in range(n):
            start = indptr[i]
            end = indptr[i+1]
            
            sigma = 0.0
            diag = 1.0
            
            # Scorrimento degli elementi non nulli della riga i
            for ptr in range(start, end):
                col = indices[ptr]
                val = data[ptr]
                
                # Somma dei contributi del triangolo inferiore
                if col < i:
                    sigma += val * x[col]
                # Salvataggio dell'elemento diagonale
                elif col == i:
                    diag = val
                    
            # Calcolo dell'incognita i-esima
            x[i] = (b[i] - sigma) / diag
            
        return x

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

        # Estrazione delle matrici triangolari L e U (A = L + U)
        L = scipy.sparse.tril(A).tocsr()
        U = scipy.sparse.triu(A, k=1).tocsr()

        # Calcolo della norma del termine noto per il criterio di arresto
        b_norm = float(np.linalg.norm(b))
        
        # Gestione del caso con b nullo
        if b_norm < np.finfo(float).eps:
            return SolverResult(x=x, iterations=0, converged=True, residual_history=[0.0], elapsed_time=0.0)

        # Inizializzazione delle variabili ausiliarie
        Ux = np.zeros(n, dtype=np.float64)
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

        # Ciclo iterativo del metodo di Gauss-Seidel
        for k in range(1, max_iter + 1):
            # Calcolo del termine noto per il sistema triangolare
            rhs = b - Ux

            # Risoluzione del sistema triangolare inferiore
            if self.use_scipy:
                x = scipy.sparse.linalg.spsolve_triangular(L, rhs, lower=True)
            else:
                x = self.forward_substitution_csr(L, rhs)

            # Calcolo del prodotto matrice-vettore
            Ux_new = U @ x

            # Aggiornamento rapido del residuo
            r = Ux - Ux_new
            Ux = Ux_new

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