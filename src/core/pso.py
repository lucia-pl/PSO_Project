from src.core.swarm import Swarm

Bounds = list[tuple[float, float]]
Vector = list[float]


class PSO:
    def __init__(
        self,
        num_particles: int,
        dim: int,
        bounds: Bounds,
        fitness_func,
        w: float,
        c1: float,
        c2: float,
        max_iters: int
    ) -> None:
        """
        Initializes the PSO optimizer.

        Args:
            num_particles (int): number of particles in the swarm
            dim (int): dimensions of the optimization problem
            bounds (Bounds): limits set for each dimension
            fitness_func: objective function
            w (float): inertia weight
            c1 (float): Cognitive/Individual coefficient
            c2 (float): Social/Collective coefficient
            max_iters (int): maximum number of iterations
        """
        self.num_particles = num_particles
        self.dim = dim
        self.bounds = bounds
        self.fitness_func = fitness_func
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iters = max_iters
        self.swarm = Swarm(num_particles, dim, bounds)

    def optimize(self) -> tuple[Vector, float]:
        """
        Runs the PSO optimization loop.
        Uses the swarm to evaluate each particle and update the global best if 
        neccesary, then updates position and velocity of all particles.

        Returns:
            tuple[Vector, float]: best position found and its fitness
        """
        s = self.swarm

        for _ in range(self.max_iters):
            s.evaluate(self.fitness_func)
            s.update(
                bounds=self.bounds,
                w=self.w,
                c1=self.c1,
                c2=self.c2
            )

        return s.global_best_position, s.global_best_fitness
