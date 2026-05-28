"""
@file        plots.py
@author      Lucía Parreño Legorburo
@brief       Creates a graph with the progress done by the particles in each function until reaching the global best
"""

import matplotlib.pyplot as plt
import os

def visualization(historic_record, show: bool, filename: str = "performance_viz"):
    """
    Graph creation

    Args:
        historic_record: all particle positions
        show (bool): open plot window
        filename (str): output filename without extension (saved to src/io/)
    """
    iterations = range(len(historic_record[0]["history"]))

    plt.figure(figsize=(8, 5))

    for entry in historic_record:
        func = entry["function"]
        history = entry["history"]
        plt.plot(iterations, history, label=func, linewidth=2)

    plt.xlabel("Iterations")
    plt.ylabel("Fitness Value")
    plt.title("PSO Convergence")

    plt.yscale("log")
    plt.grid(True)
    plt.legend()

    os.makedirs("src/io", exist_ok=True)
    out_path = os.path.join("src", "io", f"{filename}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()