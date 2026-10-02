import argparse
from pathlib import Path

# Funzioni per la gestione degli args in ingresso

def check_max_iter(value: str) -> int:
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' non è un intero valido.")

    if ivalue < 20000:
        raise argparse.ArgumentTypeError(
            f"Numero massimo di iterazioni deve essere >= 20000 (ricevuto {ivalue})."
        )
    return ivalue


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--matrix",
        metavar="PATH",
        required=True,
        type=Path,
        help="Percorso al file .mtx della matrice.",
    )
    parser.add_argument(
        "--tol",
        choices=["1e-4", "1e-6", "1e-8", "1e-10", "all"],
        metavar="FLOAT|all",
        default="all",
        help="Tolleranza di convergenza.",
    )
    parser.add_argument(
        "--max_iter",
        metavar="INT",
        type=check_max_iter,
        default=20000,
        help="Numero massimo di iterazioni.",
    )
    parser.add_argument(
        "--use_scipy",
        action="store_true",
        help="Usa la funzione scipy.sparse.linalg.spsolve_triangular per Gauss-Seidel.",
    )
    return parser