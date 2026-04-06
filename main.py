"""
@file        main.py
@author      Lucía Parreño Legorburo
@brief       File used to run the final version of the PSO with all the comparisons and best results
@date        2026-04-06
@version     1.0
"""
# NOTE: The current main only runs the basic PSO vs Pyswarm comparison, in future versions the pararelism and other
#       implementations will be added.

from experiments.run_pso import run_comparison

if __name__ == "__main__":
    run_comparison()