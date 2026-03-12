from src.core.pso import PSO


def sphere(x):
    return sum(v*v for v in x)

if __name__ == "__main__":
    dim = 2
    bounds = [(-5, 5) for _ in range(dim)]

    pso = PSO(
        num_particles=30,
        dim=dim,
        bounds=bounds,
        fitness_func=sphere,
        w=0.7,
        c1=1.5,
        c2=1.5,
        max_iters=100
    )

    best_pos, best_fit = pso.optimize()

    print("=== PSO RUN ===")
    print("Best position:", best_pos)
    print("Best fitness:", best_fit)
