"""
@file        run_benchmarks.py
@author      Lucía Parreño Legorburo
@brief       Full benchmark suite across dimensions (default: 2, 10, 30)
             Results are found in: results/benchmark_log
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
    log_dir = Path("results") / "benchmark_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = log_dir / f"benchmarks_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    logger = logging.getLogger("run_benchmarks")
    logger.info(f"Saved log: {log_file}")
    return logger


def run_for_dim(cfg: dict, dim: int, param_grid: dict, logger: logging.Logger) -> list[dict]:
    """
    Using the configuration of gridsearch values and a dimension, calls pso_gridsearch
    and extract the results for the given dimension. Then applies the logging in the best
    results of each function.

    Args:
        cfg (dict): configuration values (.yaml)
        dim (int): dimensions 
        param_grid (dict): parameters for the gridsearch
        logger (logging.Logger): previously configurated logger

    Returns:
        list[dict]: PSO results
    """
    bounds = [tuple(cfg["bounds"])] * dim
    num_particles = cfg["num_particles"]
    max_iters = cfg["max_iters"]

    logger.info("=" * 55)
    logger.info(f"DIMENSION {dim}  |  particles={num_particles}  iters={max_iters}")
    logger.info("=" * 55)

    results = pso_gridsearch(
        num_particles=num_particles,
        dim=dim,
        bounds=bounds,
        max_iters=max_iters,
        param_grid=param_grid,
    )

    logger.info(f"  {'function':<12} {'w':>6} {'c1':>6} {'c2':>6}  {'time':>8}  {'best_fitness':>14}")
    logger.info(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*6}  {'-'*8}  {'-'*14}")
    for func in FUNCTIONS:
        best = extract_best(results, func)
        logger.info(
            f"  {func:<12} {best['w']:>6.3f} {best['c1']:>6.3f} {best['c2']:>6.3f}  "
            f"{to_python_type(best['time']):>7.3f}s  {to_python_type(best['best_fitness']):>14.6e}"
        )

    return results


def main(dims: list[int], show_plot: bool):
    """
    Setups the main initializing the logger and the values for the gridsearch configurations,
    then goes through all the dimensions to extract the results and saves them in the json and csv.

    Args:
        dims (list[int]): list of posible dimensions for the trial. Default in main: 2, 10 and 30
        show_plot (bool): whether the plot shows after run. Default to false
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

    logger.info(f"Dimensions: {dims}")

    for dim in dims:
        results = run_for_dim(cfg, dim, param_grid, logger)

        plot_history = [
            {"function": func, "history": extract_best(results, func)["particle_history"]}
            for func in cfg["functions"]
        ]
        visualization(plot_history, show=show_plot, filename=f"convergence_benchmarks_dim{dim}")
        logger.info(f"Saved plot in src/io/convergence_benchmarks_dim{dim}.png")

        data_csv(results)
        data_json(results)
        logger.info(f"Saved results (dim={dim})")

    logger.info("BENCHMARK COMPLETED")

    delete_all_pycaches()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PSO benchmark suite across dimensions")
    parser.add_argument("--dims", nargs="+", type=int, default=[2, 10, 30])
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    main(dims=args.dims, show_plot=args.show)