"""
@file        V4_numpy_pso.py
@author      Lucía Parreño Legorburo
@brief       PSO version using numpy vectors
"""

import time
import numpy as np

from experiments.gridsearch import generate_grid
from src.data.objective_functions import OBJECTIVE_FUNCTIONS

Bounds = list[tuple[float, float]]
Vector = list[float]

class PSO_V4:
    """Addapts PSO and vectorizes it with numpy"""

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
    ) -> None:
        self.num_particles = num_particles
        self.dim = dim
        self.bounds = np.array(bounds)         
        self.fitness_func = fitness_func
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iters = max_iters

        self._initialize()

    def _initialize(self):
        """Initializes PSO using numpy as auxiliar"""
        low  = self.bounds[:, 0]         
        high = self.bounds[:, 1]             
        span = high - low                     

        self.positions = low + np.random.rand(self.num_particles, self.dim) * span
        self.velocities = -span + np.random.rand(self.num_particles, self.dim) * 2 * span
        self.best_pos = self.positions.copy()
        self.best_fit = np.full(self.num_particles, np.inf)

        self.global_best_position: Vector = self.positions[0].tolist()
        self.global_best_fitness: float = float("inf")
        self.global_best_history: list[float] = []


    def _evaluate_all(self) -> np.ndarray:
        """
        Evaluates all particles.
        Accepts matrix if the objective function supports them, else if
        a vector is supported apply_along_axis is used so the evaluation is row to row.

        Returns:
            np.ndarray: fitness array
        """
        try:
            fitnesses = self.fitness_func(self.positions)
            if not isinstance(fitnesses, np.ndarray) or fitnesses.shape != (self.num_particles,):
                raise ValueError("La función no devuelve un array (num_particles,)")
        except Exception:
            fitnesses = np.apply_along_axis(self.fitness_func, 1, self.positions)
        return fitnesses


    def _update_velocity(self):
        """
        Updates velocity of the particle using numpy for the randomized numbers and the array for gloabl best
        """
        r1 = np.random.rand(self.num_particles, self.dim)
        r2 = np.random.rand(self.num_particles, self.dim)

        global_best_arr = np.array(self.global_best_position)  

        cognitive = self.c1 * r1 * (self.best_pos - self.positions)
        social    = self.c2 * r2 * (global_best_arr - self.positions)

        self.velocities = self.w * self.velocities + cognitive + social

    def _update_position(self):
        """
        Updates the position using numpy clip to avoid the particle leaving the space by limiting it
        """
        self.positions += self.velocities
        low  = self.bounds[:, 0]
        high = self.bounds[:, 1]
        self.positions = np.clip(self.positions, low, high)


    def optimize(self) -> tuple[Vector, float, list]:
        """
        Evaluates all particles and updates best personal values vectorized with numpy and
        also updates global bests.

        Returns:
            tuple[Vector, float, list]: global best position, global best fitness and the global best history
        """
        for _ in range(self.max_iters):
            fitnesses = self._evaluate_all()                    

            # Actualizar mejores personales (vectorial)
            improved = fitnesses < self.best_fit
            self.best_fit  = np.where(improved, fitnesses, self.best_fit)
            self.best_pos[improved] = self.positions[improved]

            # Actualizar mejor global
            min_idx = int(np.argmin(fitnesses))
            if fitnesses[min_idx] < self.global_best_fitness:
                self.global_best_fitness = float(fitnesses[min_idx])
                self.global_best_position = self.positions[min_idx].tolist()

            self.global_best_history.append(self.global_best_fitness)

            self._update_velocity()
            self._update_position()

        return self.global_best_position, self.global_best_fitness, self.global_best_history


def pso_gridsearch_v4(
    num_particles: int,
    dim: int,
    bounds: list[tuple[float, float]],
    max_iters: int,
    param_grid: dict,
) -> list[dict]:
    """
    Gridsearch for PSO V4 (vectorized numpy).

    Args:
        num_particles, dim, bounds, max_iters: as v0
        param_grid (dict): values to create the grid to evaluate

    Returns:
        list[dict]: results using gridsearch 
    """
    grid_config = generate_grid(param_grid)
    results = []

    for func_name, func in OBJECTIVE_FUNCTIONS.items():
        for cfg in grid_config:
            start = time.time()
            pso = PSO_V4(
                num_particles=num_particles,
                dim=dim,
                bounds=bounds,
                fitness_func=func,
                w=cfg["w"],
                c1=cfg["c1"],
                c2=cfg["c2"],
                max_iters=max_iters,
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