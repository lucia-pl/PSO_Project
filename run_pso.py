"""
@file        run_pso.py
@author      Lucía Parreño Legorburo
@brief       Single PSO run: executes base PSO Vs pyswarm baseline, prints comparison,
             saves results to CSV/JSON and generates convergence plot.
"""

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from experiments.gridsearch import pso_gridsearch, baseline_gridsearch, extract_best
from src.data.print_data import print_data
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
    log_dir = Path("results") / "pso_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pso_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    logger = logging.getLogger("run_pso")
    logger.info(f"Saved log: {log_file}")
    return logger


def main(show_plot: bool = False):
    """
    With the logger and values runs both Base PSO and Pyswarm PSO to compare results.

    Args:
        show_plot (bool, optional): Whether plot is shown after run. Defaults to False.
    """
    logger = setup_logging()

    config_path = os.path.join(BASE_DIR, "experiments", "values.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    gs = cfg["gridsearch"]
    param_grid = {
        "w":  np.linspace(gs["w"]["start"],  gs["w"]["end"],  gs["w"]["steps"]).tolist(),
        "c1": np.linspace(gs["c1"]["start"], gs["c1"]["end"], gs["c1"]["steps"]).tolist(),
        "c2": np.linspace(gs["c2"]["start"], gs["c2"]["end"], gs["c2"]["steps"]).tolist(),
    }

    bounds = [tuple(cfg["bounds"])] * cfg["dim"]
    dim = cfg["dim"]
    num_particles = cfg["num_particles"]
    max_iters = cfg["max_iters"]
    functions = cfg["functions"]

    logger.info("=" * 55)
    logger.info("CONFIGURATION")
    logger.info("=" * 55)
    logger.info(f"  num_particles : {num_particles}")
    logger.info(f"  dim           : {dim}")
    logger.info(f"  bounds        : {cfg['bounds']}")
    logger.info(f"  max_iters     : {max_iters}")
    logger.info(f"  functions     : {functions}")
    logger.info("  gridsearch    :")
    for k, v in param_grid.items():
        logger.info(f"    {k}: {v}")
    logger.info("=" * 55)

    # 1. RUNS BASE PSO
    logger.info("Running base PSO...")
    pso_results = pso_gridsearch(
        num_particles=num_particles,
        dim=dim,
        bounds=bounds,
        max_iters=max_iters,
        param_grid=param_grid,
    )

    logger.info("-" * 55)
    logger.info("RESULTS of Base PSO  (Best of each Function)")
    logger.info(f"  {'function':<12} {'w':>6} {'c1':>6} {'c2':>6}  {'time':>8}  {'best_fitness':>14}")
    logger.info(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*6}  {'-'*8}  {'-'*14}")
    for func in FUNCTIONS:
        best = extract_best(pso_results, func)
        logger.info(
            f"  {func:<12} {best['w']:>6.3f} {best['c1']:>6.3f} {best['c2']:>6.3f}  "
            f"{to_python_type(best['time']):>7.3f}s  {to_python_type(best['best_fitness']):>14.6e}"
        )
    logger.info("-" * 55)

    # 2. RUNS PYSWARM BASELINE
    logger.info("Running pyswarm baseline...")
    baseline_results = baseline_gridsearch(
        bounds=bounds,
        max_iters=max_iters,
        swarmsize=num_particles,
        param_grid=param_grid,
    )
    baseline_results_list = list(baseline_results.values())

    logger.info("-" * 55)
    logger.info("RESULTS of Pyswarm baseline  (Best of each Function)")
    logger.info(f"  {'Function':<12} {'time':>8}  {'best_fitness':>14}")
    logger.info(f"  {'-'*12} {'-'*8}  {'-'*14}")
    for r in baseline_results_list:
        logger.info(
            f"  {r['function']:<12} {to_python_type(r['time']):>7.3f}s  "
            f"{to_python_type(r['best_fitness']):>14.6e}"
        )
    logger.info("-" * 55)

    # 3. COMPARES RESULTS
    print_data(pso_results, baseline_results_list)

    # 4. CREATES PLOTS AND SAVES RESULTS
    plot_history = [
        {"function": func, "history": extract_best(pso_results, func)["particle_history"]}
        for func in functions
    ]
    visualization(plot_history, show=show_plot, filename="convergence_pso")
    logger.info("Convergence plot saved in src/io/convergence_pso.png")


    data_csv(pso_results)
    data_json(pso_results)
    logger.info("Saved results in src/io/resultsCSV.csv and src/io/resultsJSON.json")
    logger.info("COMPLETED.")

    delete_all_pycaches()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run base PSO vs pyswarm baseline")
    parser.add_argument("--show", action="store_true", help="Shows plot in an independent window")
    args = parser.parse_args()
    main(show_plot=args.show)