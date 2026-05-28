"""
@file        make_viz.py
@author      Lucía Parreño Legorburo
@brief       Loads saved results from src/io/ and generates all visualisation outputs:
                1. Best fitness bar chart         in  src/io/best_fitness_bar.png
                2. Grid-search heatmap (w vs c1)  in  src/io/heatmap_<function>.png
                3. Swarm animation d=2            in  src/io/swarm_animation_2d_<function>.gif
                4. Swarm animation d=3            in  src/io/swarm_animation_3d_<function>.gif
"""

import argparse
import csv
import json
import os
import random

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from src.core.swarm import Swarm
from src.objectives.benchmarks import sphere, rastrigin, rosenbrock, ackley
from src.data.objective_functions import OBJECTIVE_FUNCTIONS  
from clean_pycaches import delete_all_pycaches

FUNCTIONS = ["sphere", "rastrigin", "rosenbrock", "ackley"]
FUNC_MAP  = OBJECTIVE_FUNCTIONS
IO_DIR    = os.path.join("src", "io")



def load_json() -> list[dict]:
    """
    Loads the results saved on the JSON

    Returns:
        list[dict]: PSO results
    """
    path = os.path.join(IO_DIR, "resultsJSON.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv() -> list[dict]:
    """
    Loads the results saved on the CSV

    Returns:
        list[dict]: PSO results
    """
    path = os.path.join(IO_DIR, "resultsCSV.csv")
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["best_fitness"] = float(row["Best_Fitness"])
            row["function"]     = row["Function"]
            row["w"]            = float(row["w"])
            row["c1"]           = float(row["c1"])
            row["c2"]           = float(row["c2"])
            rows.append(row)
    return rows




def plot_best_fitness_bar(results: list[dict], show: bool):
    """
    Best fitness bar chart

    Args:
        results (list[dict]): PSO retrieved results
        show (bool): whether it shows on screen after run
    """
    best: dict[str, float] = {}
    for r in results:
        fn = r["function"]
        bf = float(r["best_fitness"])
        if fn not in best or bf < best[fn]:
            best[fn] = bf

    funcs  = [f for f in FUNCTIONS if f in best]
    values = [best[f] for f in funcs]
    colors = ["#794CB0", "#A7803D", "#24AF59", "#C44E7D"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(funcs, values, color=colors[:len(funcs)], edgecolor="white", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_ylabel("Best Fitness (log scale)")
    ax.set_title("Best Fitness per Function  —  Base PSO")
    ax.bar_label(bars, fmt="%.2e", padding=4, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    out = os.path.join(IO_DIR, "best_fitness_bar.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[make_viz] Saved  →  {out}")
    if show:
        plt.show()
    plt.close()



def plot_heatmap(results: list[dict], show: bool):
    """
    Heatmap based on gridsearch comparing w (inertia) and c1 (cognitive coefficient)

    Args:
        results (list[dict]): PSO retrieved results
        show (bool): whether it shows on screen after run
    """
    for func in FUNCTIONS:
        func_results = [r for r in results if r["function"] == func]
        if not func_results:
            continue

        ws  = sorted(set(r["w"]  for r in func_results))
        c1s = sorted(set(r["c1"] for r in func_results))
        grid = np.full((len(c1s), len(ws)), np.nan)

        for r in func_results:
            wi  = ws.index(r["w"])
            c1i = c1s.index(r["c1"])
            val = float(r["best_fitness"])
            if np.isnan(grid[c1i, wi]) or val < grid[c1i, wi]:
                grid[c1i, wi] = val

        vmin = np.nanmin(grid)
        vmax = np.nanmax(grid)

        if vmin > 0 and vmax > 0 and vmin != vmax:
            norm = plt.matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
            cbar_label = "Best Fitness (log)"
        else:
            norm = None
            cbar_label = "Best Fitness"

        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(grid, aspect="auto", cmap="viridis_r", norm=norm)
        plt.colorbar(im, ax=ax, label=cbar_label)

        ax.set_xticks(range(len(ws)))
        ax.set_xticklabels([f"{v:.2f}" for v in ws])
        ax.set_yticks(range(len(c1s)))
        ax.set_yticklabels([f"{v:.2f}" for v in c1s])
        ax.set_xlabel("w  (inertia)")
        ax.set_ylabel("c1  (cognitive)")
        ax.set_title(f"Grid Search Heatmap for {func}  (best over c2)")

        plt.tight_layout()
        out = os.path.join(IO_DIR, f"heatmap_{func}.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"[make_viz] Saved  →  {out}")
        if show:
            plt.show()
        plt.close()


def _run_pso_capture(func, bounds, dim, num_particles, max_iters, w, c1, c2, seed):
    """Runs PSO and captures particle positions at every iteration."""
    random.seed(seed)
    swarm    = Swarm(num_particles=num_particles, dim=dim, bounds=bounds)
    history  = [] 

    for _ in range(max_iters):
        swarm.evaluate(func)
        history.append(np.array([list(p.position) for p in swarm.particles]))
        swarm.update(bounds=bounds, w=w, c1=c1, c2=c2)

    return history, swarm.global_best_position, swarm.global_best_fitness


def _contour_grid(func, bounds, resolution=200):
    """Generates a grid of values to create the contour plot for a objective function in 2D"""
    x = np.linspace(bounds[0][0], bounds[0][1], resolution)
    y = np.linspace(bounds[1][0], bounds[1][1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[func([X[i, j], Y[i, j]]) for j in range(resolution)]
                  for i in range(resolution)])
    return X, Y, Z


def animate_2d(func_name: str, show: bool,
               num_particles: int = 20, max_iters: int = 60,
               w: float = 0.5, c1: float = 1.5, c2: float = 1.5,
               seed: int = 42, fps: int = 8):
    
    """
    Creates a 2D gif animation for each objective function to show particle movement and shows depth through color
    """

    func   = FUNC_MAP[func_name]
    bounds = [(-5.12, 5.12)] * 2

    print(f"[make_viz] Running PSO for 2D animation ({func_name})...")
    history, best_pos, best_fit = _run_pso_capture(
        func, bounds, dim=2,
        num_particles=num_particles, max_iters=max_iters,
        w=w, c1=c1, c2=c2, seed=seed,
    )

    X, Y, Z = _contour_grid(func, bounds)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contourf(X, Y, Z, levels=30, cmap="viridis", alpha=0.6)
    ax.contour(X, Y, Z,  levels=30, cmap="viridis", linewidths=0.4, alpha=0.4)

    scat      = ax.scatter([], [], c="red",    s=30, zorder=5, label="Particles")
    best_scat = ax.scatter([], [], c="yellow", s=80, marker="*", zorder=6, label="Global best")
    title     = ax.set_title("")

    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.legend(loc="upper right", fontsize=8)

    def update(frame):
        """Updates animation frame to create the gif"""
        positions = history[frame]
        scat.set_offsets(positions[:, :2])
        best_scat.set_offsets([[best_pos[0], best_pos[1]]])
        title.set_text(f"{func_name}  —  iter {frame + 1}/{max_iters}  |  best={best_fit:.4e}")
        return scat, best_scat, title

    anim = animation.FuncAnimation(
        fig, update, frames=max_iters, interval=1000 // fps, blit=True
    )

    out = os.path.join(IO_DIR, f"swarm_animation_2d_{func_name}.gif")
    anim.save(out, writer="pillow", fps=fps)
    print(f"[make_viz] Saved  →  {out}")
    if show:
        plt.show()
    plt.close()


def animate_3d(func_name: str, show: bool,
               num_particles: int = 20, max_iters: int = 60,
               w: float = 0.5, c1: float = 1.5, c2: float = 1.5,
               seed: int = 42, fps: int = 8):
    """
    Creates a 3D gif animation for each objective function to show particle movement
    """
    func   = FUNC_MAP[func_name]
    bounds = [(-5.12, 5.12)] * 3

    print(f"[make_viz] Running PSO for 3D animation ({func_name})...")
    history, best_pos, best_fit = _run_pso_capture(
        func, bounds, dim=3,
        num_particles=num_particles, max_iters=max_iters,
        w=w, c1=c1, c2=c2, seed=seed,
    )

    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection="3d")

    scat      = ax.scatter([], [], [], c="red",    s=25, zorder=5, label="Particles")
    best_scat = ax.scatter([], [], [], c="yellow", s=100, marker="*", zorder=6, label="Global best")

    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_zlim(bounds[2])
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_zlabel("x₃")
    ax.legend(loc="upper left", fontsize=8)

    def update(frame):
        """Updates the frame to create the gif"""
        positions = history[frame]
        scat._offsets3d      = (positions[:, 0], positions[:, 1], positions[:, 2])
        best_scat._offsets3d = ([best_pos[0]], [best_pos[1]], [best_pos[2]])
        ax.set_title(f"{func_name} 3D  —  iter {frame + 1}/{max_iters}  |  best={best_fit:.4e}")
        return scat, best_scat

    anim = animation.FuncAnimation(
        fig, update, frames=max_iters, interval=1000 // fps, blit=False
    )

    out = os.path.join(IO_DIR, f"swarm_animation_3d_{func_name}.gif")
    anim.save(out, writer="pillow", fps=fps)
    print(f"[make_viz] Saved  →  {out}")
    if show:
        plt.show()
    plt.close()



def main(source: str, show: bool, no_anim: bool):
    """
    Main to organize all methods and run them

    Args:
        source (str): where it will read the information from (json or csv) has both to have 1 backup
        show (bool): whether the plots will show after run automatically
        no_anim (bool): whether it shows gif after run
    """
    os.makedirs(IO_DIR, exist_ok=True)

    print(f"[make_viz] Loading results from {source.upper()}...")
    results = load_json() if source == "json" else load_csv()
    print(f"[make_viz] {len(results)} entries loaded\n")

    plot_best_fitness_bar(results, show)
    plot_heatmap(results, show)

    if not no_anim:
        print()
        for func_name in FUNCTIONS:
            animate_2d(func_name, show)
            animate_3d(func_name, show)

    print("\n[make_viz] Done.")

    delete_all_pycaches()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PSO visualisation plots and animations")
    parser.add_argument("--source",  choices=["json", "csv"], default="json")
    parser.add_argument("--show",    action="store_true", help="Open each plot window")
    parser.add_argument("--no-anim", action="store_true", help="Skip GIF animations")
    args = parser.parse_args()
    main(source=args.source, show=args.show, no_anim=args.no_anim)