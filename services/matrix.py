from pathlib import Path
from typing import Union
import scipy.io
import scipy.sparse
import numpy as np

PathLike = Union[str, Path]

def load_matrix_sparse(filepath: PathLike) -> scipy.sparse.csr_matrix:
    # Carica un file .mtx e verifica che sia in formato sparso.
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Matrix file not found: {filepath}")

    matrix = scipy.io.mmread(str(filepath))
    
    # Se la matrice non è sparsa, solleviamo un errore
    if not scipy.sparse.issparse(matrix):
        raise ValueError(
            f"Errore: Il file '{filepath}' contiene una matrice densa. "
            "Questo programma supporta unicamente matrici sparse."
        )

    # Conversione in formato CSR per garantire efficienza nei calcoli
    return matrix.tocsr()
