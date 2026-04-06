from src.core.particle import Particle
from src.data.format import to_python_type

Bounds = list[tuple[float, float]]
Vector = list[float]


class Swarm:
    def __init__(self, num_particles: int, dim: int, bounds: Bounds):
        """
        Initialize swarm, the population used in the pso.

        Args:
            num_particles (int): number of particles that will compose swarms population
            dim (int): dimensions used in the optimization problem.
            bounds (Bounds): limits set for the particle in each dimension.
        """
        self.particles: list[Particle] = []

        for _ in range(num_particles):
            new_particle = Particle(dim=dim, bounds=bounds)
            self.particles.append(new_particle)

        self.global_best_position: Vector = list(self.particles[0].position)
        self.global_best_fitness: float = float("inf")
        self.global_best_history = []


    def evaluate(self, fitness_func):
        """
        Evaluates in each particle its fitness using the objective function, if the new position found is better than the global fitness
        it updates the global fitness and position.

        Args:
            fitness_func: objective function used to evaluate the particle performance
        """
        
        for p in self.particles:
            fitness = p.evaluate_particle(fitness_func)
            
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = list(p.position)

            self.global_best_history.append(self.global_best_fitness)


    def update(self, bounds: Bounds, w: float, c1: float, c2: float):
        """
        Updates velocity and position of each particle in the swarm.

        Args:
            bounds (Bounds): limits set for each dimension
            w (float): inertia weight
            c1 (float): Cognitive/Individual coefficient
            c2 (float): Social/Collective coefficient
        """
        
        for p in self.particles:
            p.update_velocity(
                global_best=self.global_best_position,
                w=w,
                c1=c1,
                c2=c2
            )
            p.update_position(bounds)
