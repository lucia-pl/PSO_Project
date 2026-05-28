"""
@file        baseline.py
@author      Lucía Parreño Legorburo
@brief       Creates the PSO using pyswarm to compare it with the base PSO
"""

from pyswarm import pso
import numpy as np
import sys
import io

from src.objectives.benchmarks import sphere, rastrigin, rosenbrock, ackley


OBJECTIVE_FUNCTIONS = {
    "sphere": sphere,
    "rastrigin": rastrigin,
    "rosenbrock": rosenbrock,
    "ackley": ackley
}


def run_baseline(
    bounds: list[tuple[float, float]],
    max_iters: int,
    swarmsize: int ,
    omega: float,
    phip: float,
    phig:float
):
    """
    Runs PSO using pyswarm library

    Args:
        bounds (list[tuple[float, float]])
        max_iters (int)
        swarmsize (int)
        omega (float): equivalent of W
        phip (float): equivalent of c1
        phig (float): equivalent of c2

    Returns:
        dict: PSO results for the best performance particle (function, best fitness and best position)
    """

    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])


    results = []

    for name, func in OBJECTIVE_FUNCTIONS.items():

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        best_pos, best_fit = pso(
            func,
            lb,
            ub,
            swarmsize=swarmsize,
            maxiter=max_iters,
            omega=omega,
            phip=phip,
            phig=phig
        )
        sys.stdout = old_stdout
        
        
        results.append({
            "function": name,
            "best_fitness": best_fit,
            "best_position": best_pos.tolist()
        })

    return results