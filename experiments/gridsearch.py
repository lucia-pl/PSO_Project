import itertools

from src.core.pso import PSO
from src.core.baseline import run_baseline
from src.objectives.benchmarks import sphere, rastrigin, rosenbrock, ackley
from src.parallel.threading.threading_pso import ThreadingPSO

OBJECTIVE_FUNCTIONS = {
    "sphere": sphere,
    "rastrigin": rastrigin,
    "rosenbrock": rosenbrock,
    "ackley": ackley
}


def generate_grid(param_cfg: dict):
    """
    Generates a matrix of all possible combinatios based on the given parameters.
    Itertools is used to create the cartesian product of the values in the given dict.
    
    Args:
        param_grid (dict): dictionary with all the values that we want to be
        used to create de grid.

    Returns:
        list[dict]: list filled of dictionaries with all the possible combination of
        the given values.
    """
    keys = list(param_cfg.keys())
    values = list(param_cfg.values())

    mix = itertools.product(*values)

    return [
        {keys[i]: m[i] for i in range(len(keys))}
        for m in mix
    ]


def pso_gridsearch(
    num_particles: int,
    dim: int,
    bounds: list[tuple[float, float]],
    max_iters: int,
    param_grid
):
    """
    Runs gridsearch for each of the functions used in the base PSO

    Args:
        num_particles (int)
        dim (int)
        bounds (list[tuple])
        max_iters (int)
        param_grid (dict): {param_name: [values]}

    Returns:
        list[dict]: gridsearch results
    """

    grid_config = generate_grid(param_grid)
    results = []

    for func_name, func in OBJECTIVE_FUNCTIONS.items():
        for cfg in grid_config:
            pso = PSO(
                num_particles=num_particles,
                dim=dim,
                bounds=bounds,
                fitness_func=func,
                w=cfg["w"],
                c1=cfg["c1"],
                c2=cfg["c2"],
                max_iters=max_iters
            )

            best_pos, best_fit, particle_history = pso.optimize()

            results.append({
                "function": func_name,
                "w": cfg["w"],
                "c1": cfg["c1"],
                "c2": cfg["c2"],
                "best_fitness": best_fit,
                "best_position": best_pos,
                "particle_history": particle_history
            })

    return results

def baseline_gridsearch(
        bounds: list[tuple[float, float]], 
        max_iters: int, 
        swarmsize: int, 
        param_grid
        ):
    """
    Runs gridsearch for each of the functions used in the pyswarm library PSO

    Args:
        bounds (list[tuple])
        max_iters (int)
        swarmsize (int)
        param_grid (dict): {param_name: [values]}

    Returns:
        list[dict]: gridsearch results
    """
    grid_config = generate_grid(param_grid)
    best_results = {name: None for name in OBJECTIVE_FUNCTIONS.keys()}

    for cfg in grid_config:
        current_results = run_baseline(
            bounds=bounds,
            max_iters=max_iters,
            swarmsize=swarmsize,
            omega = cfg["w"],
            phip=cfg["c1"],
            phig=cfg["c2"]
        )

        for res in current_results:
            name = res["function"]

            if best_results[name] is None or res["best_fitness"] < best_results[name]["best_fitness"]:
                best_results[name] = res

    return best_results

def pso_threading_gridsearch(
    num_particles: int,
    dim: int,
    bounds: list[tuple[float, float]],
    max_iters: int,
    n_threads: int,
    param_grid
    ):
    """
    Runs gridsearch for each function, used for threading PSO.

    Args:
        num_particles (int)
        dim (int)
        bounds (list[tuple])
        max_iters (int)
        n_threads (int)
        param_grid (dict)

    Returns:
        list[dict]: gridsearch results
    """

    grid_config = generate_grid(param_grid)
    results = []

    for func_name, func in OBJECTIVE_FUNCTIONS.items():
        for cfg in grid_config:

            pso = ThreadingPSO(
                num_particles=num_particles,
                dim=dim,
                bounds=bounds,
                fitness_func=func,
                w=cfg["w"],
                c1=cfg["c1"],
                c2=cfg["c2"],
                max_iters=max_iters,
                n_threads=n_threads
            )

            best_pos, best_fit, particle_history = pso.optimize()

            results.append({
                "function": func_name,
                "w": cfg["w"],
                "c1": cfg["c1"],
                "c2": cfg["c2"],
                "best_fitness": best_fit,
                "best_position": best_pos,
                "particle_history": particle_history
            })

    return results


def extract_best(results, func_name):
    """
    Obtains the best result for a objective function based on the results given by its gridsearch.

    Args:
        results (list[dict]): gridsearch results
        func_name (string)

    Returns:
        dict: the one of the list "best" with the lowest "best_fitness"
    """
    best = [r for r in results if r["function"] == func_name]
    return min(best, key=lambda x: x["best_fitness"])
