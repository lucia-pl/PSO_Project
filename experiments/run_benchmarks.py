from src.core.pso import PSO
import itertools

def sphere(x):
    return sum(v*v for v in x)

if __name__ == "__main__":
    dim = 2
    bounds = [(-5, 5) for _ in range(dim)]

    w_values = [0.4, 0.7, 1.0]
    c1_values = [1.0, 1.5, 2.0]
    c2_values = [1.0, 1.5, 2.0]

    best_config = None
    best_fitness = float("inf")

    for w, c1, c2 in itertools.product(w_values, c1_values, c2_values):
        pso = PSO(
            num_particles=30,
            dim=dim,
            bounds=bounds,
            fitness_func=sphere,
            w=w,
            c1=c1,
            c2=c2,
            max_iters=80
        )

        _, fitness = pso.optimize()

        print(f"w={w}, c1={c1}, c2={c2} → fitness={fitness}")

        if fitness < best_fitness:
            best_fitness = fitness
            best_config = (w, c1, c2)

    print("\n=== BEST CONFIGURATION ===")
    print("w, c1, c2 =", best_config)
    print("fitness =", best_fitness)
