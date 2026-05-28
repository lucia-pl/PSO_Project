"""
@file        V3_asyncio.py
@author      Lucía Parreño Legorburo
@brief       PSO version using async functions
"""

import asyncio
import random
import time

from src.core.swarm import Swarm
from experiments.gridsearch import generate_grid
from src.data.objective_functions import OBJECTIVE_FUNCTIONS

Bounds = list[tuple[float, float]]
Vector = list[float]



async def simulate_async_fitness(position: list[float], fitness_func, max_delay: float = 0.05) -> float:
    """
    Simulates an external service with randomized responses delay.

    Args:
        position: particle position
        fitness_func: objective function
        max_delay: delay in seconds

    Returns:
        float: fitness
    """
    delay = random.uniform(0, max_delay)
    await asyncio.sleep(delay)       
    return fitness_func(position)       



class SwarmAsync(Swarm):
    """
    Parent class: Swarm 
    Modifies evaluate to use asyncio
    """

    async def evaluate_async(self, async_fitness_func):
        """
        Evaluates concurrently particles with asyncio.gather()

        Args:
            async_fitness_func: helper function to recreate asyncronous scenarios
        """

        coroutines = [async_fitness_func(p.position) for p in self.particles]
        fitnesses = await asyncio.gather(*coroutines)

        for p, fitness in zip(self.particles, fitnesses):
            if fitness < p.best_fitness:
                p.best_fitness = fitness
                p.best_position = list(p.position)

            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = list(p.position)

            self.global_best_history.append(self.global_best_fitness)


class PSO_V3:
    """Addapts PSO for async evaluation"""
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
        max_delay: float = 0.05, 
    ) -> None:
        self.num_particles = num_particles
        self.dim = dim
        self.bounds = bounds
        self.fitness_func = fitness_func
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iters = max_iters
        self.max_delay = max_delay
        self.swarm = SwarmAsync(num_particles, dim, bounds)

    async def _optimize_async(self) -> tuple[Vector, float, list]:
        """
        In an async cicle each particle is evaluated an updates the swarm

        Returns:
            tuple[Vector, float, list]: global best position, global best fitness and a history of the gloal best
        """
        s = self.swarm

        async def wrapped_fitness(position):
            return await simulate_async_fitness(position, self.fitness_func, self.max_delay)

        for _ in range(self.max_iters):
            await s.evaluate_async(wrapped_fitness)
            s.update(bounds=self.bounds, w=self.w, c1=self.c1, c2=self.c2)

        return s.global_best_position, s.global_best_fitness, s.global_best_history

    def optimize(self) -> tuple[Vector, float, list]:
        """
        Helps run the async optimizer

        Returns:
            tuple[Vector, float, list]: same as _optimize_async()
        """
        return asyncio.run(self._optimize_async())


def pso_gridsearch_v3(
    num_particles: int,
    dim: int,
    bounds: list[tuple[float, float]],
    max_iters: int,
    param_grid: dict,
    max_delay: float = 0.05,
) -> list[dict]:
    """
    Gridsearch for PSO V3 (asyncio).

    Args:
        num_particles, dim, bounds, max_iters: as v0
        param_grid (dict): values to evaluate PSO
        max_delay (float): max latency in secons

    Returns:
        list[dict]: gridsearch results (pso)
    """
    grid_config = generate_grid(param_grid)
    results = []

    for func_name, func in OBJECTIVE_FUNCTIONS.items():
        for cfg in grid_config:
            start = time.time()
            pso = PSO_V3(
                num_particles=num_particles,
                dim=dim,
                bounds=bounds,
                fitness_func=func,
                w=cfg["w"],
                c1=cfg["c1"],
                c2=cfg["c2"],
                max_iters=max_iters,
                max_delay=max_delay,
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