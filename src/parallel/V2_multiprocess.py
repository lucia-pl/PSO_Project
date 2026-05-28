"""
@file        V2_multiprocess.py
@author      Lucía Parreño Legorburo
@brief       PSO version using multiprocess
"""

import time
from concurrent.futures import ProcessPoolExecutor

from src.core.swarm import Swarm
from experiments.gridsearch import generate_grid
from src.data.objective_functions import OBJECTIVE_FUNCTIONS

Bounds = list[tuple[float, float]]
Vector = list[float]



def _evaluate_batch(eval_info):
    """
    Evaluates in batches the particles in the Child Class

    Args:
        eval_info: tuple (particle list, fitness function)

    Returns:
        list[tuple]: [(fitness, best_position, best_fitness), ...]
    """
    particles_batch, fitness_func = eval_info
    results = []
    for p in particles_batch:
        fitness = p.evaluate_particle(fitness_func)
        results.append((fitness, list(p.best_position), p.best_fitness, p))
    return results


class SwarmMultiprocessing(Swarm):
    """
    Parent class: Swarm 
    Modifies evaluate to using multiple processes and batching
    """

    def evaluate(self, fitness_func, executor: ProcessPoolExecutor, batch_size: int = 1):
        """
        Evaluates particles in separated processes grouped in batches, 
        and reuses the executor to avoid creating and destroying processes.

        Args:
            fitness_func: objective function
            executor: ProcessPoolExecutor previously initialized
            batch_size: particles per process (bigger value = less paralelism)
        """
        batches = [
            self.particles[i: i + batch_size]
            for i in range(0, len(self.particles), batch_size)
        ]
        tasks = [(batch, fitness_func) for batch in batches]

        all_batch_results = list(executor.map(_evaluate_batch, tasks))

        # Reuse of processes
        idx = 0
        for batch_results in all_batch_results:
            for fitness, best_pos, best_fit, updated_p in batch_results:
                p = self.particles[idx]
                p.position = list(updated_p.position)
                p.velocity = list(updated_p.velocity)
                p.best_position = best_pos
                p.best_fitness = best_fit

                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = list(p.position)

                self.global_best_history.append(self.global_best_fitness)
                idx += 1


class PSO_V2:
    """Addapts the PSO to the multiprocess optimization"""
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
        batch_size: int = 1,
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
        self.batch_size = batch_size
        self.swarm = SwarmMultiprocessing(num_particles, dim, bounds)

    def optimize(self) -> tuple[Vector, float, list]:
        """
        Creates a ProcessPoolExecutor used to evaluate the fitness of each particle and updates velocity and position
        this happen in batches of particles.

        Returns:
            tuple[Vector, float, list]: returns best position, globa best fitness and the history of the global best
        """
        s = self.swarm
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            for _ in range(self.max_iters):
                s.evaluate(
                    self.fitness_func,
                    executor=executor,
                    batch_size=self.batch_size,
                )
                s.update(bounds=self.bounds, w=self.w, c1=self.c1, c2=self.c2)
        return s.global_best_position, s.global_best_fitness, s.global_best_history


def pso_gridsearch_v2(
    num_particles: int,
    dim: int,
    bounds: list[tuple[float, float]],
    max_iters: int,
    param_grid: dict,
    max_workers: int = None,
    batch_size: int = 4,
) -> list[dict]:
    """
    Gridsearch for PSO V2 (multiprocessing + batching).

    Args:
        num_particles, dim, bounds, max_iters: like v0
        param_grid (dict): values to evaluate
        max_workers (int): parallel processes (None = cpu_count)
        batch_size (int): particles per process

    Returns:
        list[dict]: gridsearch (pso) results
    """
    grid_config = generate_grid(param_grid)
    results = []

    for func_name, func in OBJECTIVE_FUNCTIONS.items():
        for cfg in grid_config:
            start = time.time()
            pso = PSO_V2(
                num_particles=num_particles,
                dim=dim,
                bounds=bounds,
                fitness_func=func,
                w=cfg["w"],
                c1=cfg["c1"],
                c2=cfg["c2"],
                max_iters=max_iters,
                max_workers=max_workers,
                batch_size=batch_size,
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