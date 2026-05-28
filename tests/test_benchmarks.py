"""
@file        test_benchmarks.py
@author      Lucía Parreño Legorburo
@brief       Tests for the objective functions in src/objectives/benchmarks.py
"""

from math import isclose
from src.objectives.benchmarks import sphere, rastrigin, ackley, rosenbrock


class TestSphere:
    """Tests for objective function: Sphere"""
    def test_global_minimum(self):
        """Sphere minimum is 0 at the origin"""
        assert sphere([0.0, 0.0, 0.0]) == 0.0

    def test_known_value(self):
        """sphere([1, 2, 3]) = 1 + 4 + 9 = 14"""
        assert sphere([1.0, 2.0, 3.0]) == 14.0

    def test_always_non_negative(self):
        """Sphere is always >= 0"""
        assert sphere([-3.0, 2.5, -1.1]) >= 0.0

    def test_single_dimension(self):
        assert sphere([4.0]) == 16.0


class TestRastrigin:
    """Tests for objective function: Rastrigin"""
    def test_global_minimum(self):
        """Rastrigin minimum is 0 at the origin"""
        assert isclose(rastrigin([0.0, 0.0]), 0.0, abs_tol=1e-10)

    def test_always_non_negative(self):
        """Rastrigin is always >= 0"""
        assert rastrigin([-2.5, 1.3, -0.7]) >= 0.0

    def test_single_dimension(self):
        assert isclose(rastrigin([0.0]), 0.0, abs_tol=1e-10)

    def test_scales_with_dimension(self):
        """Value at origin for any d should be 0"""
        for d in [2, 5, 10]:
            assert isclose(rastrigin([0.0] * d), 0.0, abs_tol=1e-10)


class TestAckley:
    """Tests for objective function: Ackley"""
    def test_global_minimum(self):
        """Ackley minimum is ~0 at the origin"""
        assert isclose(ackley([0.0, 0.0]), 0.0, abs_tol=1e-10)

    def test_always_non_negative(self):
        assert ackley([1.5, -2.0]) >= 0.0

    def test_empty_vector(self):
        """Empty vector should return 0"""
        assert ackley([]) == 0.0

    def test_scales_with_dimension(self):
        for d in [2, 5, 10]:
            assert isclose(ackley([0.0] * d), 0.0, abs_tol=1e-10)


class TestRosenbrock:
    """Tests for objective function: Rosenbrock"""
    def test_global_minimum(self):
        """Rosenbrock minimum is 0 at [1, 1, ..., 1]"""
        assert isclose(rosenbrock([1.0, 1.0]), 0.0, abs_tol=1e-10)

    def test_known_value_2d(self):
        """rosenbrock([0, 0]) = 100*(0-0)^2 + (1-0)^2 = 1"""
        assert isclose(rosenbrock([0.0, 0.0]), 1.0, abs_tol=1e-10)

    def test_always_non_negative(self):
        assert rosenbrock([-1.0, 2.0, 0.5]) >= 0.0

    def test_minimum_nd(self):
        """Minimum is 0 at allones vector for any dimension"""
        for d in [2, 5, 10]:
            assert isclose(rosenbrock([1.0] * d), 0.0, abs_tol=1e-10)