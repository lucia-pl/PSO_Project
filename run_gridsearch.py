"""
@file        run_gridsearch.py
@author      Lucía Parreño Legorburo
@brief       Hyperparameter grid search over (w, c1, c2).
             Results are found in: results/gridsearch_log
"""

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from experiments.gridsearch import pso_gridsearch, extract_best
from src.data.save_data import data_csv, data_json
from src.data.format import to_python_type
from src.viz.plots import visualization
from clean_pycaches import delete_all_pycaches

BASE_DIR  = os.path.dirname(__file__)
FUNCTIONS = ["sphere", "rastrigin", "rosenbrock", "ackley"]


def setup_logging() -> logging.Logger:
    """
    Configurations for the logging such as where it will be stored, the name and format of shown results,
    instead of overwriting the log, logs will be identified by day and run hour to saved them.
    """
    log_dir = Path("results") / "gridsearch_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"grid_search_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    logger = logging.getLogger("run_grid_search")
    logger.info(f"Saved log: {log_file}")
    return logger


def log_top_results(logger: logging.Logger, results: list[dict], func_name: str, top_n: int):
    """
    Using the logger shows the top (amount: top_n) results of each objective function.

    Args:
        logger (logging.Logger): configurated logger
        results (list[dict]): PSO results
        func_name (str): name of the objective function
        top_n (int): amount of top results shown. Defaults in main to 3
    """
    func_results = sorted(
        [r for r in results if r["function"] == func_name],
        key=lambda r: r["best_fitness"]
    )
    logger.info("-" * 55)
    logger.info(f"TOP {top_n} of {func_name}")
    logger.info(f"  {'rank':>4} {'w':>6} {'c1':>6} {'c2':>6}  {'time':>8}  {'best_fitness':>14}")
    logger.info(f"  {'----':>4} {'-'*6} {'-'*6} {'-'*6}  {'-'*8}  {'-'*14}")
    for rank, r in enumerate(func_results[:top_n], start=1):
        logger.info(
            f"  {rank:>4} {r['w']:>6.3f} {r['c1']:>6.3f} {r['c2']:>6.3f}"
            f"{to_python_type(r['time']):>7.3f}s  {to_python_type(r['best_fitness']):>14.6e}"
        )
    logger.info("-" * 55)


def main(top_n: int, show_plot: bool):
    """
    Setups the logger and gridsearch values obtained from values.yaml,
    and shows the top (top_n) results of each function, creates the convergence plot and saves
    both results and logging results.

    Args:
        top_n (int): amount of top results shown for each function
        show_plot (bool): whether the plot is shown after run. Defaults to false
    """
    logger = setup_logging()

    config_path = os.path.join(BASE_DIR, "experiments", "values.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    gs = cfg["gridsearch"]
    param_grid = {
        "w": np.linspace(gs["w"]["start"],  gs["w"]["end"],  gs["w"]["steps"]).tolist(),
        "c1": np.linspace(gs["c1"]["start"], gs["c1"]["end"], gs["c1"]["steps"]).tolist(),
        "c2": np.linspace(gs["c2"]["start"], gs["c2"]["end"], gs["c2"]["steps"]).tolist(),
    }

    bounds = [tuple(cfg["bounds"])] * cfg["dim"]
    dim = cfg["dim"]
    num_particles = cfg["num_particles"]
    max_iters = cfg["max_iters"]
    functions = cfg["functions"]

    total = gs["w"]["steps"] * gs["c1"]["steps"] * gs["c2"]["steps"]

    logger.info("=" * 55)
    logger.info("GRID SEARCH CONFIGURATION")
    logger.info("=" * 55)
    logger.info(f"  combinations : {total}")
    logger.info(f"  functions     : {functions}")
    logger.info(f"  num_particles : {num_particles}")
    logger.info(f"  dim           : {dim}")
    logger.info(f"  max_iters     : {max_iters}")
    logger.info("  gridsearch    :")
    for k, v in param_grid.items():
        logger.info(f"    {k}: {v}")
    logger.info("=" * 55)

    logger.info("Running grid search...")
    results = pso_gridsearch(
        num_particles=num_particles,
        dim=dim,
        bounds=bounds,
        max_iters=max_iters,
        param_grid=param_grid,
    )

    for func in FUNCTIONS:
        log_top_results(logger, results, func, top_n)

    plot_history = [
        {"function": func, "history": extract_best(results, func)["particle_history"]}
        for func in functions
    ]
    visualization(plot_history, show=show_plot, filename="convergence_gridsearch")
    logger.info("Saved plot in src/io/convergence_gridsearch.png")

    data_csv(results)
    data_json(results)
    logger.info("Saved results in src/io/resultsCSV.csv and src/io/resultsJSON.json")
    logger.info("GRID SEARCH COMPLETED.")

    delete_all_pycaches()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PSO hyperparameter grid search")
    parser.add_argument("--top",  type=int, default=3)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    main(top_n=args.top, show_plot=args.show)