"""
@file        benchmarks.py
@author      Lucía Parreño Legorburo
@brief       Commonly used objective functions used in the trials and evaluation of PSO
"""

from math import cos, pi, exp, sqrt
Vector = list[float]


def sphere(x: Vector) -> float:
    """
    Continuous, convex and unimodal (single maximun and minimum) benchmark function.

    f(x) = sum(x_i ^ 2)
    """
    return sum(xi**2 for xi in x)


def rastrigin(x: Vector) -> float:
    """
    Non-convex, multimodal (multiple local minimums) function used to test performance in optimization.

    f(x) = 10*d + sum(x_i^2 - 10*cos*(2*pi*x_i))
    """
    d = len(x)
    return (10.0*d + sum(xi*xi - 10.0*cos(2.0*pi*xi) for xi in x))


def ackley(x: Vector) -> float:
    """
    Non-convex, highly multimodal with many local minimums and usually one "deep" minimum.

    f(x) = -20 * exp(-0.2 * sqrt(1/n * sum(x_i^2)))
            - exp(1/n * sum(cos(2*pi*x_i)))
            + 20 + e
    """

    n = len(x)
    if n == 0:
        return 0.0
    
    sum_sq = sum(xi*xi for xi in x)
    sum_cos =  sum(cos(2.0*pi*xi) for xi in x)

    sq_part = -20.0 * exp(-0.2*sqrt(sum_sq/n))
    cos_part = -exp(sum_cos/n)

    return sq_part + cos_part + 20.0 + exp(1.0)


def rosenbrock(x: Vector) -> float:
    """
    Non-convex, unimodal function which minimum lies inside a parabolic-shaped flat and narrow valley.
    Commonly used in gradient-based optimization.

    f(x,y) = (a-x)^2 + b*(y-x^2)^2    [for 2D]
    f(x) = f(x_1, x_2, ..., x_n) = sum(100*(x_{i+1}-x_i^2)^2 + (1-x_i)^2) for i = 0...n-2  [Multidimensions]

    """
    n = len(x)-1
    result = 0.0

    for i in range(n):
        xi = x[i]
        xj = x[i+1]
        result += (100.0 * (xj-xi*xi)**2 + (1.0-xi)**2)

    return result    