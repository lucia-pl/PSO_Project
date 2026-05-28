"""
@file        run_parallels.py
@author      Lucía Parreño Legorburo
@brief       Runs all parallel PSO versions and save results to compare their performance
             Results are found in: results/parallels_log
"""

import os
import logging
import yaml
from datetime import datetime
from pathlib import Path

from src.parallel.V1_threading import pso_gridsearch_v1
from src.parallel.V2_multiprocess import pso_gridsearch_v2
from src.parallel.V3_asyncio import pso_gridsearch_v3
from src.parallel.V4_numpy_pso import pso_gridsearch_v4
from clean_pycaches import delete_all_pycaches


def setup_logging() -> logging.Logger:
    """
    Configurations for the logging such as where it will be stored, the name and format of shown results,
    instead of overwriting the log, logs will be identified by day and run hour to saved them.
    """
    log_dir = Path("results") / "parallels_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = log_dir / f"parallels_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    logger = logging.getLogger("pso_benchmark")
    logger.info(f"Saved log: {log_file}")
    return logger


def load_config(path: str = "experiments/values.yaml") -> dict:
    """
    Load values to configure gridsearch and other PSO values

    Args:
        path (str, optional): route to values.yaml . Defaults to "experiments/values.yaml".

    Returns:
        dict: configuration values
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_param_grid(cfg: dict) -> dict:
    """
    Using the values for configuration creates the grid parameters

    Args:
        cfg (dict): configuration values

    Returns:
        dict: gridsearch values (w, c1, c2)
    """
    def linspace(start, end, steps):
        if steps == 1:
            return [start]
        step = (end - start) / (steps - 1)
        return [round(start + i * step, 6) for i in range(steps)]

    gs = cfg["gridsearch"]
    return {
        "w":  linspace(gs["w"]["start"],  gs["w"]["end"],  gs["w"]["steps"]),
        "c1": linspace(gs["c1"]["start"], gs["c1"]["end"], gs["c1"]["steps"]),
        "c2": linspace(gs["c2"]["start"], gs["c2"]["end"], gs["c2"]["steps"]),
    }


def log_config(logger: logging.Logger, cfg: dict, derived: dict):
    """
    Builds logging structure that will be given to the user and added to the final log,
    it shows the configuration values of the experiment.

    Args:
        logger (logging.Logger): logger configuration
        cfg (dict): configuration (PSO) values
        derived (dict): specific values for parallel PSO versions
    """
    logger.info("=" * 55)
    logger.info("CONFIGURATION")
    logger.info("=" * 55)
    logger.info(f"  num_particles : {cfg['num_particles']}")
    logger.info(f"  dim           : {cfg['dim']}")
    logger.info(f"  bounds        : {cfg['bounds']}")
    logger.info(f"  max_iters     : {cfg['max_iters']}")
    logger.info(f"  functions     : {cfg.get('functions', [])}")
    logger.info("  gridsearch    :")
    for key, vals in derived["param_grid"].items():
        logger.info(f"    {key}: {vals}")
    logger.info("  workers       :")
    logger.info(f"    V1 workers   : {derived['v1_workers']}")
    logger.info(f"    V2 workers   : {derived['v2_workers']}")
    logger.info(f"    V2 batch     : {derived['v2_batch']}")
    logger.info(f"    V3 max_delay : {derived['v3_max_delay']}")
    logger.info("=" * 55)


def log_results(logger: logging.Logger, version_name: str, results: list[dict]):
    """
    Builds logging structure that will be given to the user and added to the final log,
    it shows the configuration values of the experiment and is used individually in each
    PSO version.

    Args:
        logger (logging.Logger): logger configuration
        version_name (str): PSO version used
        results (list[dict]): results for the version
    """
    best_per_func: dict[str, dict] = {}
    for r in results:
        fn = r["function"]
        if fn not in best_per_func or r["best_fitness"] < best_per_func[fn]["best_fitness"]:
            best_per_func[fn] = r

    best_runs = list(best_per_func.values())

    # 1. Establish the version shown and headers for the columns
    logger.info("-" * 55)
    logger.info(f"RESULTS for {version_name}  (Best run for each function)")
    logger.info(f"  {'function':<12} {'w':>6} {'c1':>6} {'c2':>6}  {'time':>8}  {'best_fitness':>14}")
    logger.info(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*6}  {'-'*8}  {'-'*14}")

    # 2. Finds the best result for each function
    for r in best_runs:
        logger.info(
            f"  {r['function']:<12} "
            f"{r['w']:>6.3f} "
            f"{r['c1']:>6.3f} "
            f"{r['c2']:>6.3f}  "
            f"{r['time']:>7.3f}s  "
            f"{r['best_fitness']:>14.6e}"
        )

    # 3. Gives a summary of the total
    total_time = sum(r["time"] for r in results)
    best_overall = min(r["best_fitness"] for r in best_runs)
    logger.info(f"  {'─'*55}")
    logger.info(f"  total_time    : {total_time:.3f}s  (all runs)")
    logger.info(f"  n_runs        : {len(results)}")
    logger.info(f"  best_overall  : {best_overall:.6e}")
    logger.info("-" * 55)


def main():
    """
    Setups the main running all parallel versions, showing and saving the results
    """

    # 1. Log and Values configuration
    logger = setup_logging()
    cfg = load_config()

    num_particles = cfg["num_particles"]
    dim = cfg["dim"]
    bounds_raw = cfg["bounds"]
    bounds = [(bounds_raw[0], bounds_raw[1])] * dim
    max_iters = cfg["max_iters"]
    param_grid = build_param_grid(cfg)

    workers = cfg.get("workers", {})
    v1_workers = workers.get("v1_workers", None)
    v2_workers = workers.get("v2_workers", None)
    v2_batch = workers.get("v2_batch") or max(1, num_particles // (os.cpu_count() or 4))
    v3_max_delay = workers.get("v3_max_delay", 0.02)

    log_config(logger, cfg, {
        "param_grid": param_grid,
        "v1_workers": v1_workers,
        "v2_workers": v2_workers,
        "v2_batch": v2_batch,
        "v3_max_delay": v3_max_delay,
    })

    # 2. Runs V1, Threading
    logger.info("Running V1 (threading)...")
    res_v1 = pso_gridsearch_v1(
        num_particles, dim, bounds, max_iters, param_grid,
        max_workers=v1_workers,
    )
    log_results(logger, "V1 threading", res_v1)

    # 3. Runs V2, Multiprocessing
    logger.info("Running V2 (multiprocessing)...")
    res_v2 = pso_gridsearch_v2(
        num_particles, dim, bounds, max_iters, param_grid,
        max_workers=v2_workers,
        batch_size=v2_batch,
    )
    log_results(logger, "V2 multiprocessing", res_v2)

    # 4. Runs V3, Asyncio
    logger.info("Running V3 (asyncio, simulated latency)...")
    res_v3 = pso_gridsearch_v3(
        num_particles, dim, bounds, max_iters, param_grid,
        max_delay=v3_max_delay,
    )
    log_results(logger, "V3 asyncio", res_v3)

    # 5. Runs V4, Vectorized NumPy
    logger.info("Running V4 (Vectorized NumPy)...")
    res_v4 = pso_gridsearch_v4(
        num_particles, dim, bounds, max_iters, param_grid,
    )
    log_results(logger, "V4 numpy", res_v4)

    logger.info("ALL PARALLEL VERSIONS COMPLETED.")

    delete_all_pycaches()


if __name__ == "__main__":
    main()