from __future__ import annotations

from abc import ABC, abstractmethod
from collections import namedtuple

import numpy as np
import scipy.sparse
import scipy.sparse.linalg


SolverResult = namedtuple(
    "SolverResult",
    ["x", "iterations", "converged", "residual_history", "elapsed_time"],
)


class BaseSolver(ABC):
    @property
    def name(self) -> str:
        return self.__class__.__name__.replace("Solver", "")

    @abstractmethod
    def solve(
        self,
        A: scipy.sparse.spmatrix,
        b: np.ndarray,
        tol: float,
        max_iter: int,
    ) -> SolverResult:
        """Risolve il sistema lineare **Ax = b** iterativamente."""


    @staticmethod
    def validate(A: scipy.sparse.spmatrix, b: np.ndarray, mat_iter: int) -> None:
        """Verifica la sparsità, la quadratura, la simmetria, la positività definita della matrice e il numero di iterazioni."""
        
        # Verifica del numero massimo di iterazioni
        if mat_iter < 20000:
            raise ValueError(f"Il numero massimo di iterazioni deve essere almeno 20000 (ricevuto {mat_iter}).")

        # Verifica della sparsità
        if not scipy.sparse.issparse(A):
            raise TypeError("Errore di sistema: L'algoritmo accetta unicamente matrici sparse di tipo scipy.sparse.")

        m, n = A.shape

        # Verifica che la matrice sia quadrata
        if m != n:
            raise ValueError(f"La matrice A deve essere quadrata (formato attuale: {A.shape}).")

        # Verifica compatibilità del termine noto
        if b.shape[0] != m:
            raise ValueError(f"Il vettore b deve avere dimensione pari alla matrice (A: {m}, b: {b.shape[0]}).")

        # Verifica della simmetria
        diff = A - A.T
        if diff.nnz > 0 and np.max(np.abs(diff.data)) > 1e-10:
            raise ValueError("La matrice A non è simmetrica.")

        # Verifica della positività definita
        try:
            A_float = A.astype(float)
            # Per matrici piccole usiamo Cholesky (molto affidabile)
            if n <= 500:
                np.linalg.cholesky(A_float.toarray())
            # Per matrici grandi calcoliamo l'autovalore più piccolo
            else:
                min_eig = scipy.sparse.linalg.eigsh(A_float, k=1, which="SA", return_eigenvectors=False)[0]
                if min_eig <= 1e-10:
                    raise ValueError("La matrice A non è definita positiva.")

        except np.linalg.LinAlgError:
            raise ValueError("La matrice A non è definita positiva.")
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Impossibile verificare la positività definita: {e}") from e

    

