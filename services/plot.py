import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import scipy.sparse
from pathlib import Path

def plot_histograms(results_data: list[dict], matrix_path: Path):
    matrix_name = matrix_path.stem
    solvers = ["Jacobi", "GaussSeidel", "Gradient", "ConjugateGradient"]
    # Estrai le tolleranze uniche dai dati e ordinale in modo decrescente
    tolerances = sorted(list(set(r["tol"] for r in results_data)), reverse=True)
    
    # Crea la cartella plots se non esiste
    plot_dir = Path("plots")
    plot_dir.mkdir(exist_ok=True)
    
    x = np.arange(len(tolerances))
    width = 0.2
    
    colors = {
        "Jacobi": "#1f77b4",
        "GaussSeidel": "#ff7f0e",
        "Gradient": "#2ca02c",
        "ConjugateGradient": "#d62728"
    }
    
    metrics = ["elapsed_time", "iterations", "err_rel"]
    titles = ["Tempo di Esecuzione", "Numero di Iterazioni", "Errore Relativo"]
    y_labels = ["Secondi (log)", "Iterazioni", "Errore (log)"]
    filenames = ["time", "iterations", "error"]
    
    for i, metric in enumerate(metrics):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        is_log = metric != "iterations"
        
        for j, solver in enumerate(solvers):
            y = []
            for t in tolerances:
                point = next((r for r in results_data if r["solver"] == solver and r["tol"] == t), None)
                val = point[metric] if point else (1e-16 if is_log else 0)
                y.append(val)
            
            offset = (j - 1.5) * width
            rects = ax.bar(x + offset, y, width, label=solver, log=is_log, color=colors[solver])
            
            if metric == "iterations":
                ax.bar_label(rects, padding=3, fmt='%g', fontsize=8)
            else:
                ax.bar_label(rects, padding=3, fmt='%.2e', fontsize=8)
            
        ax.set_title(f"{titles[i]} - Matrice: {matrix_name}", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{t:.0e}" for t in tolerances])
        ax.set_xlabel("Tolleranza")
        ax.set_ylabel(y_labels[i])
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.3)
        plt.tight_layout()
        
        out_path = plot_dir / f"{matrix_name}_{filenames[i]}.png"
        plt.savefig(out_path, dpi=300)
        print(f"[INFO] Grafico salvato: {out_path}")
        plt.close(fig)

