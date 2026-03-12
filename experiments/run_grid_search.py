from src.core.pso import PSO
import math

def sphere(x):
    return sum(v*v for v in x)

def rastrigin(x):
    return 10*len(x) + sum(v*v - 10*math.cos(2*math.pi*v) for v in x)

def rosenbrock(x):
    return sum(100*(x[i+1]-x[i]**2)**2 + (1-x[i])**2 for i in range(len(x)-1))

benchmarks = {
    "Sphere": sphere,
    "Rastrigin": rastrigin,
    "Rosenbrock": rosenbrock
}

if __name__ == "__main__":
    dim = 2
    bounds = [(-5, 5) for _ in range(dim)]

    for name, func in benchmarks.items():
        pso = PSO(
            num_particles=30,
            dim=dim,
            bounds=bounds,
            fitness_func=func,
            w=0.7,
            c1=1.5,
            c2=1.5,
            max_iters=100
        )

        best_pos, best_fit = pso.optimize()

        print(f"\n=== {name} ===")
        print("Best position:", best_pos)
        print("Best fitness:", best_fit)
