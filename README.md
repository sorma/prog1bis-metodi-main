# Progetto Metodi del Calcolo Scientifico — Progetto 1

Questo progetto implementa una libreria per la risoluzione di sistemi lineari $Ax = b$ tramite metodi iterativi, ottimizzata per operare su matrici sparse (formato `.mtx`).

Vengono implementati e confrontati i seguenti algoritmi:
1. **Metodo di Jacobi**
2. **Metodo di Gauss-Seidel** (con opzione per solutore SciPy o custom)
3. **Metodo del Gradiente** (Steepest Descent)
4. **Metodo del Gradiente Coniugato**

## Documentazione

Una descrizione dettagliata del progetto, inclusa l'analisi dei risultati e i dettagli implementativi, è disponibile all'interno della directory `docs/`. In particolare, consultare `Relazione_MDS.pdf` e il file sorgente `relazione.tex`.

## Autori

- **Andrea Mauri** — 945327
- **Luca Rossini** — 900230
- **Francesco Sormani** — 894552

Corso di Laurea Magistrale in Informatica  
Università degli Studi di Milano-Bicocca  
A.A. 2025-2026

---

## Setup del progetto

### 1. Clona la repository

```bash
git clone https://github.com/Payd3r/prog1bis-metodi.git
cd prog1bis-metodi
```

### 2. Setup del virtual environment

Aprire la directory del progetto nel terminale ed eseguire:

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux/macOS
```

### 3. Installa le dipendenze

Dopo aver attivato il virtual environment è possibile installare le dipendenze.

```bash
pip install -r requirements.txt
```  

---

# Run

Per eseguire i test sui metodi iterativi, avviare il file `main.py`. È possibile specificare la matrice da utilizzare, la tolleranza e il numero massimo di iterazioni.

```bash
python main.py --matrix matrix/spa1.mtx --tol all --max_iter 20000
```  

### Parametri disponibili:
- `--matrix PATH`: Percorso al file `.mtx` della matrice da analizzare.
- `--tol FLOAT|all`: Tolleranza di convergenza (es. `1e-6`). Se impostato su `all` (default), il test verrà eseguito per le tolleranze $10^{-4}, 10^{-6}, 10^{-8}, 10^{-10}$.
- `--max_iter INT`: Numero massimo di iterazioni consentite (default: 20000).
- `--use_scipy`: Flag per utilizzare la funzione `scipy.sparse.linalg.spsolve_triangular` per il metodo di Gauss-Seidel (per confronto di performance).

Al termine dell'esecuzione verranno mostrati a console i dati relativi a errore relativo, numero di iterazioni e tempo impiegato. Inoltre, verranno generati dei grafici comparativi all'interno della cartella `plots/`.

---
