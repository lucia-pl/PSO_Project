"""
@file        V1_threading.py
@author      Lucía Parreño Legorburo
@brief       PSO version using threads
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.core.swarm import Swarm
from experiments.gridsearch import generate_grid 
from src.data.objective_functions import OBJECTIVE_FUNCTIONS  

Bounds = list[tuple[float, float]]
Vector = list[float]


class SwarmThreading(Swarm):
    """
    Parent class: Swarm 
    Modifies evaluate to use threads
    """

    def evaluate(self, fitness_func, max_workers: int = None):
        """
        Evaluates independently (different threads) each particle

        Args:
            fitness_func: objective function
            max_workers: number of threads (if its None it uses os.cpu_count())
        """
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(p.evaluate_particle, fitness_func): p
                for p in self.particles
            }
            for future in as_completed(futures):
                fitness = future.result()
                p = futures[future]
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = list(p.position)
                self.global_best_history.append(self.global_best_fitness)


class PSO_V1:
    """PSO adapted with the evaluation using threads"""
    def __init__(
        self,
        num_particles: int,
        dim: int,
        bounds: Bounds,
        fitness_func,
        w: float,
        c1: float,
        c2: float,
        max_iters: int,
        max_workers: int = None,
    ) -> None:
        self.num_particles = num_particles
        self.dim = dim
        self.bounds = bounds
        self.fitness_func = fitness_func
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iters = max_iters
        self.max_workers = max_workers
        self.swarm = SwarmThreading(num_particles, dim, bounds)

    def optimize(self) -> tuple[Vector, float, list]:
        s = self.swarm
        for _ in range(self.max_iters):
            s.evaluate(self.fitness_func, max_workers=self.max_workers)
            s.update(bounds=self.bounds, w=self.w, c1=self.c1, c2=self.c2)
        return s.global_best_position, s.global_best_fitness, s.global_best_history


def pso_gridsearch_v1(
    num_particles: int,
    dim: int,
    bounds: list[tuple[float, float]],
    max_iters: int,
    param_grid: dict,
    max_workers: int = None,
) -> list[dict]:
    """
    Gridsearch for PSO V1 (threading).

    Args:
        num_particles, dim, bounds, max_iters: as base PSO (v0)
        param_grid (dict): given parameters from saved file 
        max_workers (int): amount of threads (None = predetermined)

    Returns:
        list[dict]: gridsearch results as v0
    """
    grid_config = generate_grid(param_grid)
    results = []

    for func_name, func in OBJECTIVE_FUNCTIONS.items():
        for cfg in grid_config:
            start = time.time()
            pso = PSO_V1(
                num_particles=num_particles,
                dim=dim,
                bounds=bounds,
                fitness_func=func,
                w=cfg["w"],
                c1=cfg["c1"],
                c2=cfg["c2"],
                max_iters=max_iters,
                max_workers=max_workers,
            )
            best_pos, best_fit, history = pso.optimize()
            elapsed = time.time() - start

            results.append({
                "time": elapsed,
                "function": func_name,
                "w": cfg["w"],
                "c1": cfg["c1"],
                "c2": cfg["c2"],
                "best_fitness": best_fit,
                "best_position": best_pos,
                "particle_history": history,
            })

    return results