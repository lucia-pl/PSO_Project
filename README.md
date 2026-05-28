# PSO - Parallel & Concurrent Programming

Implementation of Particle Swarm Optimization (PSO) in Python to compare parallel and concurrent programming strategies and their efficiency in different scenarios. This project includes grid-search, result persistence, visualization and a use case using a real-world scenario.

---

## Project Structure

```
PSO_ProgramacionParalela/
├── src/
│   ├── core/               # PSO engine: PSO, Swarm, Particle
│   ├── parallel/           # Parallel versions: V1, V2, V3, V4
│   ├── objectives/         # Benchmark functions + SVM objective
│   ├── data/               # Persistence, formatting, printing
│   ├── viz/                # Convergence plots
│   └── io/                 # Output files (CSV, JSON, PNG)
├── experiments/
│   ├── gridsearch.py       # Grid search logic
│   └── values.yaml         # Global configuration
├── tests/                  # Unit tests (pytest)
├── results/                # Execution logs per script
├── run_pso.py              # Run base PSO vs pyswarm baseline
├── run_benchmarks.py       # Benchmark suite across dimensions
├── run_grid_search.py      # Hyperparameter grid search
├── run_parallels.py        # Compare parallel versions V1–V4
├── make_viz.py             # Generate plots and animations
├── use_case.py             # Real-world use case: SVM on Digits
└── requirements.txt
```

---

## Architecture

The project uses a modular design where the parallel versions of the PSO remain the same and the evaluation strategy is modified based on the different parallelism methods.

### Module dependency diagram

```
experiments/values.yaml
        │
        │
  run_*.py / use_case.py          ← entry points
        │
        ├── experiments/gridsearch.py
        │           │
        │           │
        │     src/core/pso.py          ← orchestrates one PSO run
        │           │
        │           │
        │     src/core/swarm.py        ← manages the particle population
        │           │
        │           ├── src/core/particle.py     ← position, velocity, bounds
        │           │
        │           └── src/parallel/            ← parallelism evaluator
        │                   ├── parallel_evaluator.py
        │                   ├── V1_threading.py
        │                   ├── V2_multiprocess.py
        │                   ├── V3_asyncio.py
        │                   └── V4_numpy_pso.py
        │
        ├── src/objectives/           ← fitness functions (pure callables)
        │       ├── benchmarks.py      (sphere, rastrigin, rosenbrock, ackley)
        │       └── svm_benchmark.py   (SVMObjective, TrackedSVMObjective)
        │
        ├── src/data/                 ← persistence and display
        │       ├── save_data.py       (CSV, JSON)
        │       ├── print_data.py      (PrettyTable)
        │       └── format.py          (numpy to Python types)
        │
        └── src/viz/plots.py          ← convergence plots
```

### Design decisions

**Core and Evaluator**

There is a single core for the PSO and the evaluator is swapped depending on the PSO version that is being used. The inmutable core consists of `PSO`, `Swarm` and `Particle` while Swarm.evaluate() chages across V0 (non-parallel) and the different parallel versions V1-V4, its diference is how it destributes the fitness calls.

There is an abstract class, `FitnessEvaluator`, used to adapt the different evauation methods, it is located in `src/parallel/parallel_evaluator.py`. The parallel subclasses adjust `evaluate()` to its necessities while keeping the other parts identical.

**Bounds and Clamp**

When particles leave the search space, its position is clamped to the nearest boundary, this security method to ensure no particles go out of bounds is implemented in `Particle.update_position()`. The strategy was chosen for its simplicity and to avoid artificial fitness distorsions near boundaries, this way the particles still shows that it was going to the outer bounds without leaving the space.

**Topology**

The global best position is a standard position shared between all particles and represented in `Swarm.global_best_position`. This is based on a fully-connected topology also named star topology, it converges fast however in multimodal functions it can cause premature convergencia.

**Fitness Functions**

The objective functions are simple and based on a given list of floats and a returned float, the simplicity helps with the implementation for the other versions. In the case of V2 (multiprocessing) the functions must be picklable (serialization).

---

## Installation

**Requirements:** Python 3.10+

```bash
# 1. Clone the repository
git clone <repo-url>
cd PSO_ProgramacionParalela

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate.bat       # Windows CMD
# source venv/bin/activate       # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Configuration

All global parameters are defined in `experiments/values.yaml`.
Different configurations have been used during the tests and trials and the main objective of keeping the modifiable values in a different file is keeping the code clean and allow users to easily modify parameters for their own experiments.

---

## Usage

### Run base PSO vs pyswarm baseline
```bash
python run_pso.py
python run_pso.py --show        # open plot window
```
Outputs: `src/io/convergence_pso.png`, `src/io/resultsCSV.csv`, `src/io/resultsJSON.json`

---

### Run benchmark suite across dimensions
```bash
python run_benchmarks.py
python run_benchmarks.py --dims 2 10 30 --show
```
Outputs: `src/io/convergence_benchmarks_dim{N}.png` for each dimension

---

### Run hyperparameter grid search
```bash
python run_grid_search.py
python run_grid_search.py --top 5       # show top-5 configs per function
```
Outputs: `src/io/convergence_gridsearch.png`, results appended to CSV/JSON

---

### Compare parallel versions (V1–V4)
```bash
python run_parallels.py
```
Outputs: best result per function per version in `results/parallels_log/`

---

### Generate visualisation plots and animations
```bash
python make_viz.py
python make_viz.py --no-anim            # skip GIF animations (faster)
python make_viz.py --source csv         # load from CSV instead of JSON
```
Outputs:
- `src/io/best_fitness_bar.png`
- `src/io/heatmap_{function}.png`
- `src/io/swarm_animation_2d_{function}.gif`
- `src/io/swarm_animation_3d_{function}.gif`

---

### Run real-world use case: SVM on Digits
```bash
python use_case.py
python use_case.py --cv 5 --seed 42 --show
```
Optimises SVM hyperparameters (C, gamma) on the sklearn Digits dataset using PSO V0–V3 and compares against GridSearchCV.

Outputs:
- `src/io/use_case_exploration.png`   ← search space explored by each version
- `src/io/use_case_comparison.png`    ← accuracy and time comparison
- `src/io/use_case_speedup.png`       ← speedup of V1/V2/V3 vs V0

---

### Run tests
```bash
python -m pytest tests/ -v
```

---

## Parallel Strategies

| Version | Strategy | Key characteristic |
|---|---|---|
| V0 | Sequential | Baseline, no parallelism |
| V1 | Threading (`ThreadPoolExecutor`) | Shared memory, GIL-limited for pure Python; effective when fitness calls release the GIL (e.g. NumPy, sklearn) |
| V2 | Multiprocessing (`ProcessPoolExecutor`) | True parallelism, IPC overhead, requires picklable fitness functions |
| V3 | Asyncio (`asyncio.gather`) | Cooperative concurrency, suited for I/O-bound or latency-simulated workloads |
| V4 | NumPy vectorisation | Implicit parallelism via BLAS, eliminates Python loop overhead entirely |

---

## Benchmark Functions

| Function | Global minimum | Characteristics |
|---|---|---|
| Sphere | 0 at origin | Convex, unimodal, easy baseline |
| Rastrigin | 0 at origin | Multimodal, many local minima |
| Rosenbrock | 0 at [1,...,1] | Narrow valley, hard to navigate |
| Ackley | 0 at origin | Highly multimodal, deceptive landscape |

---

## Logs

Each script writes timestamped logs to its own subfolder under `results/` in order to create a single file for each trial or experiment in a clean and readable way:

```
results/
├── usecase_log/        ← use_case.py
├── pso_log/            ← run_pso.py
├── benchmark_log/      ← run_benchmarks.py
├── gridsearch_log/     ← run_grid_search.py
└── parallels_log/      ← run_parallels.py
```

---