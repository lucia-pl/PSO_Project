from src.core.pso import PSO
from src.parallel.threading.threading_swarm import ThreadingSwarm

Bounds = list[tuple[float, float]]

class ThreadingPSO(PSO):
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
        n_threads=4
    ):
        """
        Initialization of herediraty PSO class: Threading version.
        The major change is the update of Swarm, using ThreadingSwarm to implement
        parallelization and the number of threads that will be used on the Swarm.

        Args:
            num_particles (int)
            dim (int)
            bounds (Bounds)
            fitness_func
            w (float)
            c1 (float)
            c2 (float)
            max_iters (int)
            n_threads (int, optional): Number of threads used to divide the process. Defaults to 4.
        """

        super().__init__(
            num_particles,
            dim,
            bounds,
            fitness_func,
            w,
            c1,
            c2,
            max_iters
        )
        self.swarm = ThreadingSwarm(num_particles, dim, bounds, n_threads)
