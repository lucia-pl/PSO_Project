from concurrent.futures import ThreadPoolExecutor
from src.core.swarm import Swarm

Bounds = list[tuple[float, float]]

class ThreadingSwarm(Swarm):
    def __init__(self, num_particles: int, dim: int, bounds: Bounds, n_threads: int=4):
        """
        Initialization of hereditary class Swarm: Threading version.

        Args:
            num_particles (int): particles used in the swarm (population)
            dim (int): number of dimensions
            bounds (Bounds): limits set for the particle in each dimension
            n_threads (int, optional): Number of threads used to divide the process. Defaults to 4.
        """
        super().__init__(num_particles, dim, bounds)
        self.n_threads = n_threads

    def evaluate(self, fitness_func):
        """
        Parallel evaluation of particles using ThreadPoolExecutor.
        The evaluation of the particle will be executed in parallel and then
        the global best will be updated sequentially.

        Args:
            fitness_func: objective function used for evaluation of performance.
        """

        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            fitness_values = list(
                executor.map(
                    lambda p: p.evaluate_particle(fitness_func),
                    self.particles
                )
            )

        for p, fitness in zip(self.particles, fitness_values):
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = list(p.position)

            self.global_best_history.append(self.global_best_fitness)
