from pathlib import Path

def print_header(matrix_path: Path, n: int, tol: float, max_iter: int) -> None:
    sep = "=" * 81
    print(sep)
    print(
        f"Matrix: {matrix_path.name}  |  n={n}"
    )
    print(f"tol={tol:.2e}  |  max_iter={max_iter}")
    print(sep)
    print(
        f"{'Metodo':<20} | {'Err.Rel.':<10} | "
        f"{'Iter':<5} | {'Tempo(s)':<9} | {'Stato'}"
    )
    print("-" * 81)


def print_row(
    method_name: str,
    err_rel: float,
    iterations: int,
    elapsed: float,
    converged: bool,
) -> None:
    stato = "OK" if converged else "NO-CONV"
    print(
        f"{method_name:<20} | {err_rel:<10.2e} | "
        f"{iterations:<5} | {elapsed:<9.4f} | {stato}"
    )