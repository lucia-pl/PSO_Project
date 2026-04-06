"""
@file        run_threading.py
@author      Lucía Parreño Legorburo
@brief       Runner for the threading version of the PSO
@date        2026-04-06
@version     1.0
"""
# NOTE: Currently this version is only able to run through terminal and does not save the results 
#       To run use: python -m experiments.run_threading
# TODO: + saving results, + comparison with base pso, + show it on main 

import time
import numpy as np
import yaml
import os


from experiments.gridsearch import pso_threading_gridsearch, baseline_gridsearch
from src.data.print_data import print_data


BASE_DIR = os.path.dirname(__file__)

def run_comparison_threading():
    """
    Function used to run the Threading version of PSO, it is similar to the run of the base PSO
    but using the threading versions.
    """
    
    config_path = os.path.join(BASE_DIR, "values.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    functions = cfg["functions"]
    gs = cfg["gridsearch"]

    param_grid = {
        "w": np.linspace(gs["w"]["start"], gs["w"]["end"], gs["w"]["steps"]),
        "c1": np.linspace(gs["c1"]["start"], gs["c1"]["end"], gs["c1"]["steps"]),
        "c2": np.linspace(gs["c2"]["start"], gs["c2"]["end"], gs["c2"]["steps"]),
        "functions": functions
    }

    bounds = [(cfg["bounds"])] * cfg["dim"]
    dim = cfg["dim"]
    num_particles = cfg["num_particles"]
    max_iters = cfg["max_iters"]
    n_threads = cfg["n_threads"] 

    # PSO THREADING
    start_pso = time.time()
    pso_results = pso_threading_gridsearch(
        num_particles=num_particles,
        dim=dim,
        bounds=bounds,
        max_iters=max_iters,
        param_grid=param_grid,
        n_threads=n_threads
    )
    end_pso = time.time()
    pso_time = end_pso - start_pso

    # BASELINE (pyswarm)
    start_base = time.time()
    baseline_results = baseline_gridsearch(
        bounds=bounds,
        max_iters=max_iters,
        swarmsize=num_particles,
        param_grid=param_grid
    )
    end_base = time.time()
    baseline_time = end_base - start_base

    baseline_results_list = list(baseline_results.values())

    print_data(pso_results, baseline_results_list, pso_time, baseline_time)


if __name__ == "__main__":
    run_comparison_threading()