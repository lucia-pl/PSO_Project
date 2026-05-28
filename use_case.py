"""
@file        use_case.py
@author      Lucía Parreño Legorburo
@brief       Case study where it compares V0 (sequential), V1 (threading), V2 (multiprocessing), V3 (asyncio)
             and sklearn GridSearchCV as external baseline.
             Results are found in: results/usecase_log

@details
Case study: SVM hyperparameter optimisation on the Digits dataset using PSO.

C (margin of separation in classes) and gamma (influence ratio),
if C is low it will be more tolerant, while being high means it is strict,
if gamma is low values will be more global, while being high means values are more local.
The search is done using a log10 space and each PSO version minimises 1 - cross_val_accuracy over (log10_C, log10_gamma).

Studied dataset: The dataset has 1797 8x8 images of hand-written digits, to use them they are previously transformed to
vector on lenght 64.
"""

import argparse
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.core.pso import PSO
from src.parallel.V1_threading import PSO_V1
from src.parallel.V2_multiprocess import PSO_V2
from src.parallel.V3_asyncio import PSO_V3
from src.objectives.svm_benchmark import make_svm_objective, make_tracked_svm_objective, SVM_BOUNDS
from src.data.format import to_python_type
from clean_pycaches import delete_all_pycaches

IO_DIR = os.path.join("src", "io")

PSO_KWARGS = dict(num_particles=20, dim=2, bounds=SVM_BOUNDS,
                  w=0.6, c1=1.5, c2=1.5, max_iters=40)


def setup_logging() -> logging.Logger:
    """
    Configurations for the logging such as where it will be stored, the name and format of shown results,
    instead of overwriting the log, logs will be identified by day and run hour to save them.
    """
    log_dir = Path("results") / "usecase_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = log_dir / f"use_case_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    logger = logging.getLogger("use_case")
    logger.info(f"Saved log in: {log_file}")
    return logger



def _run(label: str, pso_instance, fitness_func,
         logger: logging.Logger) -> dict:
    """
    Configurations to run all versions of PSO and show their extracted information.

    Args:
        label (str): name of the parallelism version
        pso_instance: PSO
        fitness_func: objective function
        logger (logging.Logger): logging configuration

    Returns:
        dict: version, best C, best gamma, accuracy distributing in classes, time it took to distribute,
                how many combinations where evaluated.
    """
    t0 = time.perf_counter()
    best_pos, best_fit, history = pso_instance.optimize()
    elapsed = time.perf_counter() - t0

    C_best     = 10 ** best_pos[0]
    gamma_best = 10 ** best_pos[1]
    accuracy   = 1.0 - best_fit
    ep = getattr(fitness_func, "evaluated_points", [])

    logger.info("-" * 50)
    logger.info(f"RESULTS {label}")
    logger.info(f"  C (classes separation)  : {C_best:.6f}")
    logger.info(f"  gamma (influence ratio) : {gamma_best:.6f}")
    logger.info(f"  accuracy                : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    logger.info(f"  evaluations             : {len(ep)}")
    logger.info(f"  time                    : {elapsed:.2f}s")
    logger.info("-" * 50)

    return {"label": label, "C": C_best, "gamma": gamma_best,
            "accuracy": accuracy, "time": elapsed,
            "evaluated_points": list(ep)}


def run_all_versions(X, y, cv: int, seed: int, logger: logging.Logger) -> list[dict]:
    """
    Runs all parallel PSO versions (except 4) and version 0 (base PSO) and saves them in a 
    list to use later to compare with the baseline.

    Version 4 is not used as it uses a vectorizable objective function and SVM does not apply in this case,
    if it is forced then instead of parallelism a sequential loop occurs which defeats the purpose of V4.

    Args:
        X : features of the dataset
        y : classes of the dataset
        cv (int): cross validation (amount of evaluations of C and gamma in different partitions)
        seed (int): reproducible seed
        logger (logging.Logger): logger configuration

    Returns:
        list[dict]: all versions results
    """
    results = []

    # V0 (Sequential)
    random.seed(seed)
    fitness_func = make_tracked_svm_objective(X, y, cv)
    results.append(_run("V0 Sequential", PSO(**PSO_KWARGS, fitness_func=fitness_func),
                        fitness_func, logger))

    # V1 (Threading)
    random.seed(seed)
    fitness_func = make_tracked_svm_objective(X, y, cv)
    results.append(_run("V1 Threading", PSO_V1(**PSO_KWARGS, fitness_func=fitness_func),
                        fitness_func, logger))

    # V2 (Multiprocessing (SVMObjective: picklable, no tracking))
    # V2 does not track so it will consider it did 0 evaluations
    random.seed(seed)
    fitness_func = make_svm_objective(X, y, cv)
    results.append(_run("V2 Multiprocessing",
                        PSO_V2(**PSO_KWARGS, fitness_func=fitness_func, batch_size=4),
                        fitness_func, logger))

    # V3 (Asyncio)
    random.seed(seed)
    fitness_func = make_tracked_svm_objective(X, y, cv)
    results.append(_run("V3 Asyncio",
                        PSO_V3(**PSO_KWARGS, fitness_func=fitness_func, max_delay=0.02),
                        fitness_func, logger))

    return results


def run_gridsearch_baseline(X, y, cv: int, logger: logging.Logger) -> dict:
    """
    Baseline used to compare with V0 and the other parallel PSO versions.

    Args:
        X : features of the dataset
        y : classes of the dataset
        cv (int): cross validation (amount of evaluations of C and gamma in different partitions)
        logger (logging.Logger): configured logger

    Returns:
        dict: Baseline results
    """
    param_grid = {
        "svm__C":     [0.01, 0.1, 1, 10, 100],
        "svm__gamma": [0.001, 0.01, 0.1, 1],
    }
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("svm",    SVC(kernel="rbf", random_state=42))])

    logger.info("Running GridSearchCV as baseline...")
    t0 = time.perf_counter()
    gs = GridSearchCV(pipe, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
    gs.fit(X, y)
    elapsed = time.perf_counter() - t0

    best_params   = gs.best_params_
    best_accuracy = gs.best_score_

    logger.info("-" * 50)
    logger.info("RESULTS for GridSearchCV")
    logger.info(f"  C (classes separation) : {best_params['svm__C']}")
    logger.info(f"  gamma (influence ratio) : {best_params['svm__gamma']}")
    logger.info(f"  accuracy   : {best_accuracy:.4f}  ({best_accuracy*100:.2f}%)")
    logger.info(f"  time       : {elapsed:.2f}s")
    logger.info(f"  combined evaluations: {len(gs.cv_results_['params'])}")
    logger.info("-" * 50)

    return {"label": "GridSearchCV", "C": best_params["svm__C"],
            "gamma": best_params["svm__gamma"],
            "accuracy": best_accuracy, "time": elapsed}



def plot_exploration(pso_results: list[dict], show: bool):
    """
    One scatter subplot per PSO version showing explored (C, gamma) space.
    Color = cross validation accuracy 
    Red star = best point found

    Args:
        pso_results (list[dict]): results obtained from the PSO parallel and V0 versions
        show (bool): whether a window with the plot opens after run
    """

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()

    # 1. Creates a plot for each exploration
    for ax, r in zip(axes, pso_results):
        pts      = r["evaluated_points"]
        log_c    = [p[0] for p in pts]
        log_g    = [p[1] for p in pts]
        accuracy = [p[2] for p in pts]

        sc = ax.scatter(log_c, log_g, c=accuracy, cmap="viridis",
                        s=35, alpha=0.7, edgecolors="none", vmin=0, vmax=1)
        plt.colorbar(sc, ax=ax, label="CV Accuracy")
        ax.scatter([np.log10(r["C"])], [np.log10(r["gamma"])],
                   c="red", s=180, marker="*", zorder=5,
                   label=f"Best acc={r['accuracy']:.4f}")
        ax.set_xlabel("log_10(C)")
        ax.set_ylabel("log_10(gamma)")
        ax.set_title(f"{r['label']}  —  {len(pts)} evals  |  {r['time']:.1f}s")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.3)

    # 2. Save results
    fig.suptitle("PSO Search Exploration (SVM Hyperparameter Optimisation on Digits)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    out = os.path.join(IO_DIR, "use_case_exploration.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[use_case] Saved in {out}")
    if show:
        plt.show()
    plt.close()


def plot_comparison(pso_results: list[dict], gs_result: dict, show: bool):
    """
    Bar charts comparing accuracy and time across all versions and GridSearchCV.

    Args:
        pso_results (list[dict]): parallel and V0 results
        gs_result (dict): gridsearch (baseline) results
        show (bool): whether a window with the plot opens after run
    """
    
    all_results = pso_results + [gs_result]
    labels   = [r["label"] for r in all_results]
    accuracy = [r["accuracy"] for r in all_results]
    times    = [r["time"] for r in all_results]
    colors   = ["#344F7C", "#4C925C", "#873A3C", "#7867B1", "#6F1953"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. ACCURACY COMPARISON
    bars = axes[0].bar(labels, accuracy, color=colors, edgecolor="white")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Cross-Validation Accuracy")
    axes[0].set_title("Accuracy comparison")
    axes[0].bar_label(bars, fmt="%.4f", padding=4, fontsize=8)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].tick_params(axis="x", rotation=15)

    # 2. EXECUTION TIME COMPARISON
    bars2 = axes[1].bar(labels, times, color=colors, edgecolor="white")
    axes[1].set_ylabel("Time (s)")
    axes[1].set_title("Execution time comparison")
    axes[1].bar_label(bars2, fmt="%.2fs", padding=4, fontsize=8)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    axes[1].tick_params(axis="x", rotation=15)

    # 3. Save results
    fig.suptitle("PSO versions vs GridSearchCV (SVM on Digits)", fontsize=12)
    plt.tight_layout()
    out = os.path.join(IO_DIR, "use_case_comparison.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[use_case] Saved in {out}")
    if show:
        plt.show()
    plt.close()


def plot_speedup(pso_results: list[dict], show: bool):
    """
    Speedup of V1, V2, V3 relative to V0 baseline.
    Compares times to find the fastest version for this use case.

    Args:
        pso_results (list[dict]): parallel versions and V0 results
        show (bool): whether a window with the plot shows after run
    """
    # 1. Establishes V0 as base for speedup comparison
    v0_time = next(r["time"] for r in pso_results if "V0" in r["label"])
    labels  = [r["label"] for r in pso_results if "V0" not in r["label"]]
    speedup = [v0_time / r["time"] for r in pso_results if "V0" not in r["label"]]
    colors  = ["#55A868", "#C44E52", "#8172B2"]

    # 2. Creates plot comparing with parallel versions
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, speedup, color=colors, edgecolor="white")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="V0 baseline (1x)")
    ax.set_ylabel("Speedup vs V0")
    ax.set_title("Parallel Speedup (SVM Hyperparameter Optimisation on Digits)")
    ax.bar_label(bars, fmt="%.2fx", padding=4)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    # 3. Saves results
    out = os.path.join(IO_DIR, "use_case_speedup.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[use_case] Saved in {out}")
    if show:
        plt.show()
    plt.close()


def main(cv: int, seed: int, show: bool):
    """
    Sets the main where all methods are used to create the whole use case: pso, baseline and plots
    for comparison.

    Args:
        cv (int): cross validation. Defaults to 5 
        seed (int): reproducible seed
        show (bool): whether the plot shows after run. Defaults to false
    """
    logger = setup_logging()
    os.makedirs(IO_DIR, exist_ok=True)

    digits = load_digits()
    X, y   = digits.data, digits.target

    logger.info("=" * 50)
    logger.info("USE CASE: SVM optimization using PSO")
    logger.info("Dataset: Digits (sklearn)")
    logger.info(f"  samples  : {X.shape[0]}")
    logger.info(f"  features  : {X.shape[1]}")
    logger.info(f"  classes    : {len(set(y))}  {to_python_type(sorted(set(y)))}")
    logger.info(f"  cv folds  : {cv}")
    logger.info(f"  seed      : {seed}")
    logger.info("=" * 50)

    # 1. Runs PSO versions
    pso_results = run_all_versions(X, y, cv=cv, seed=seed, logger=logger)

    # 2. Runs GridSearchCV baseline
    gs_result = run_gridsearch_baseline(X, y, cv=cv, logger=logger)

    # 3. Creates the summary
    logger.info("=" * 50)
    logger.info("SUMMARY")
    logger.info(f"  {'Version':<22} {'Accuracy':>10}  {'Time':>8}")
    logger.info(f"  {'-'*22} {'-'*10}  {'-'*8}")
    for r in pso_results:
        logger.info(f"  {r['label']:<22} {r['accuracy']:>10.4f}  {r['time']:>7.2f}s")
    logger.info(f"  {gs_result['label']:<22} {gs_result['accuracy']:>10.4f}  {gs_result['time']:>7.2f}s")
    v0_time = next(r["time"] for r in pso_results if "V0" in r["label"])
    for r in pso_results:
        if "V0" not in r["label"]:
            logger.info(f"  Speedup {r['label']}: {v0_time/r['time']:.2f}x")
    logger.info("=" * 50)

    # 4. Creates the plots
    plot_exploration(pso_results, show)
    plot_comparison(pso_results, gs_result, show)
    plot_speedup(pso_results, show)

    logger.info("USE CASE COMPLETED.")

    delete_all_pycaches()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PSO use case: SVM on Digits")
    parser.add_argument("--cv",   type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    main(cv=args.cv, seed=args.seed, show=args.show)