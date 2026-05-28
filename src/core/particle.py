"""
@file        particle.py
@author      Lucía Parreño Legorburo
@brief       Initialization of particle and all the methods to evaluate its performance and movement around the given trials space
"""

import random

Bounds = list[tuple[float, float]]
Vector = list[float]


class Particle:
    def __init__(self, dim: int, bounds: Bounds) -> None:
        """
        Class instantiation.
        - position: random position bounded by the limits declared (bounds), each dimension will have a number randomly generated. 
        For dimension "i" there will be a number generated between bounds[i][0] and bounds[i][1]
        - velocity: initial random velocity delimited by the search space size. The size will be a calculated value, 
        then a random velocity is generated using the boundaries calculated.
        - best_postition: the initial postion will be saved as the best one at the moment and later updated after 
        finding another one better.
        - best_fitness: result of the best postion found in all iterations, will only be changed after a better position is found.
        Initialized as infinite so any first value found is automatically better.

        Args:
            dim (int): dimmensions in which the particle will be optimized
            bounds (Bounds): boundaries allowed for each dimension (min,max)
        """
        self.position: Vector = []
        self.velocity: Vector = []
        self.best_position: Vector = list(self.position)
        self.best_fitness = float("inf")
        self.initialize(dim=dim, bounds=bounds)
        

    def initialize(self, dim:int, bounds: Bounds) -> None:
        for i in range(dim):
            boundaries = abs(bounds[i][1]-bounds[i][0])
            value_vel = random.uniform(-boundaries, boundaries)
            self.velocity.append(value_vel) 

        for i in range(dim):
            min_bound = bounds[i][0]
            max_bound = bounds[i][1]
            value_pos = random.uniform(min_bound,max_bound)
            self.position.append(value_pos)


    def evaluate_particle(self, fitness_func: object) -> float:
        """
        Uses the current position to evaluate its performance using fitness_func, 
        if the position returned is better than the current fitness registered the fitness is 
        updated to the value calculated with the current position and fitness_func.

        Args:
            fitness_func (object): objective function used for the optimization

        Returns:
            float: current fitness found in this iteration (even if it is not the best)
        """
        fitness: float = fitness_func(self.position)

        if fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_position = list(self.position)

        return fitness

    def update_velocity(
        self,
        global_best: Vector,
        w: float,
        c1: float,
        c2: float
    ) -> None:
        """
        Updates velocity for each dimension, in order to keep randomness two numbers between 0 and 1 are generated
        to avoid particles acting the same way. uses c1 (cognitive) and c2 (social) to calculate the distances from 
        personal and global best positions.

        Args:
            global_best (Vector): best global postion (in the PSO) to time
            w (float): inertia weight (if w is bigger it will be faster)
            c1 (float): Cognitive/Individual coefficient. Higher c1 makes particles keep closer to its best position.
            c2 (float): Social/Collective coefficient. Higher c2 makes particles close faster to the best GLOBAL position.
        """
        for i in range(len(self.position)):
            r1 = random.random()
            r2 = random.random()

            cognitive = c1 * r1 * (self.best_position[i] - self.position[i])
            social = c2 * r2 * (global_best[i] - self.position[i])

            self.velocity[i] = w * self.velocity[i] + cognitive + social

    def update_position(self, bounds: Bounds) -> None:
        """
        Updates the position of the particle using its velocity while making sure the boundaries are not trespassed.
        Going through all the dimensions: the particle moves in the dimension with the calculated velocity, the limits
        are suppervised and if they go out of them the new postion will be reduced to the bound value.

        Args:
            bounds (Bounds): limits for each dimension
        """
        for i in range(len(self.position)):
            self.position[i] += self.velocity[i]

            if self.position[i] < bounds[i][0]:
                self.position[i] = bounds[i][0]
            elif self.position[i] > bounds[i][1]:
                self.position[i] = bounds[i][1]

