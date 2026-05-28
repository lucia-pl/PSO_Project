"""
@file        print_data.py
@author      Lucía Parreño Legorburo
@brief       Creates tables for final comparisons and easy reading of the results of the trials
"""

from prettytable import PrettyTable
from experiments.gridsearch import extract_best
from src.data.format import to_python_type

FUNCTIONS = ["sphere", "rastrigin", "rosenbrock", "ackley"]



def print_data(pso_results, baseline_results):
    """
    Prints data in a readable and clear way.
    Extracts the best results of each type of pso for each function and formats the information
    to make it easy to read and compare.
    Shows comparison between the created PSO and pyswarm pso, then finds the best results for 
    the PSO for each function to show the values used to reach that solution.

    Args:
        pso_results (list[dict])
        baseline_results (list[dict])
        pso_time (float)
        baseline_time (float)
    """

    print("\n\n===  FINAL COMPARISON ===")

    table_compare = PrettyTable()
    table_compare.field_names = [
        "Function",
        "PSO Best Fitness",
        "PSO Time (s)",
        "Baseline Best Fitness",
        "Baseline Time (s)"
    ]

    for func in FUNCTIONS:
        best_pso = extract_best(pso_results, func)
        best_base = next(r for r in baseline_results if r["function"] == func)

        table_compare.add_row([
            func,
            to_python_type(best_pso["best_fitness"]),
            to_python_type(best_pso["time"]),
            to_python_type(best_base["best_fitness"]),
            to_python_type(best_base["time"]),
        ])

    print(table_compare)

    print("\n=== Best configuration for PSO ===")

    table_pso = PrettyTable()
    table_pso.field_names = ["Function", "w", "c1", "c2", "Fitness"]

    for func in FUNCTIONS:
        best = extract_best(pso_results, func)
        table_pso.add_row([
            func,
            best["w"],
            best["c1"],
            best["c2"],
            to_python_type(best["best_fitness"])
        ])

    print(table_pso)

