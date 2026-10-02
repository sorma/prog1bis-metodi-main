#python main.py --matrix matrix/spa1.mtx --tol 1e-6 --max_iter 20000

from __future__ import annotations
from services.parser import build_parser
from pathlib import Path
import numpy as np
from linear_iterative_solvers import (
    ConjugateGradientSolver,
    GaussSeidelSolver,
    GradientSolver,
    JacobiSolver
)
from services.print import (
    print_header,
    print_row
)
from services.matrix import (
    load_matrix_sparse
)
from services.utils import (
    make_b,
    run_solver,
    relative_error
)
from services.plot import plot_histograms


def main():
    # Costruzione parser
    parser = build_parser()
    args = parser.parse_args()

    # Definizione dei risolutori
    solvers = [
        JacobiSolver(),
        GaussSeidelSolver(use_scipy=args.use_scipy),
        GradientSolver(),
        ConjugateGradientSolver(),
    ]

    # Costruzione lista tolleranze
    if args.tol == "all": 
        tolerances: list[float] = [1e-4, 1e-6, 1e-8, 1e-10]
    else: 
        tolerances: list[float] = [float(args.tol)]

    # Caricamento matrice sparsa
    matrix_path: Path = args.matrix
    print(f"\nCaricamento matrice: {matrix_path} ...")
    try:
        A = load_matrix_sparse(matrix_path)
    except Exception as e:
        print(f"\n[ERRORE]: {e}")
        return

    # Dimensione del sistema lineare
    n = A.shape[0]
    # Costruzione x_exact e b
    b, x_exact = make_b(A)

    # Inizializza la lista per salvare i risultati dei test per i grafici
    results_data = []

    # Esecuzione dei metodi iterativi
    for tol in tolerances:
        print_header(matrix_path, n, tol, args.max_iter)
        for solver in solvers:
            metrics = run_solver(solver, A, b, tol=tol, max_iter=args.max_iter)
            err_rel = relative_error(metrics["x"], x_exact)
            
            # Salva i dati per i grafici
            results_data.append({
                "solver": solver.name,
                "tol": tol,
                "err_rel": err_rel,
                "iterations": metrics["iterations"],
                "elapsed_time": metrics["elapsed_time"]
            })
            
            print_row(
                    method_name=solver.name,
                    err_rel=err_rel,
                    iterations=metrics["iterations"],
                    elapsed=metrics["elapsed_time"],
                    converged=metrics["converged"],
                )

        print("=" * 81)
        
    # Genera gli istogrammi
    if results_data:
        plot_histograms(results_data, matrix_path)


if __name__ == "__main__":
    main()