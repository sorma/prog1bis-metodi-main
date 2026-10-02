# Linear Iterative Solvers
Libreria Python per la risoluzione iterativa di sistemi lineari sparsi (Ax=b).

### Come usarla:
1. Installa in modalità sviluppo: `pip install -e .`
2. Importa un risolutore: `from linear_iterative_solvers import ConjugateGradientSolver`
3. Risolvi il sistema: `result = ConjugateGradientSolver().solve(A, b, tol=1e-6, max_iter=20000)`
4. Leggi i dati: Accedi alla soluzione con `result.x` e al numero di iterazioni con `result.iterations`.
