"""
@file        parallel_evaluator.py
@author      Lucía Parreño Legorburo
@brief       Common base evaluator for all parallel PSO versions
"""
from abc import ABC, abstractmethod
from typing import Callable

Vector = list[float]


class FitnessEvaluator(ABC):
    @abstractmethod
    def evaluate(
        self,
        positions: list[Vector],
        fitness_func: Callable[[Vector], float],
    ) -> list[float]:
        """
        Base for all Child Classes,
        evaluate fitness for a batch of particle positions.

        Args:
            positions: List of position vectors, shape (n_particles, dim).
            fitness_func: Objective function.

        Returns:
            List of fitness values, same length and order as positions.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"