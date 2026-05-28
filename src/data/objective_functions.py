"""
@file        objective_functions.py
@author      Lucía Parreño Legorburo
@brief       Groups all the functions to use in the different PSO runs 
                in order to have them in one place for easy access and modification
"""

from src.objectives.benchmarks import sphere, rastrigin, rosenbrock, ackley

OBJECTIVE_FUNCTIONS = {
    "sphere": sphere,
    "rastrigin": rastrigin,
    "rosenbrock": rosenbrock,
    "ackley": ackley
}