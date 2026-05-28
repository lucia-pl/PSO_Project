"""
@file        test_pso.py
@author      Lucía Parreño Legorburo
@brief       Tests for src/core/pso.py integration level tests
"""

import random
from src.core.pso import PSO
from src.objectives.benchmarks import sphere, rastrigin


class TestPSOConvergence:
    """Tests for PSO convergence"""
    def test_sphere_converges_near_zero(self):
        random.seed(42)
        pso = PSO(
            num_particles=30,
            dim=5,
            bounds=[(-5.12, 5.12)] * 5,
            fitness_func=sphere,
            w=0.5,
            c1=1.5,
            c2=1.5,
            max_iters=200,
        )
        _, best_fitness, _ = pso.optimize()
        assert best_fitness < 1.0, f"Expected fitness < 1.0, got {best_fitness}"

    def test_returns_three_values(self):
        random.seed(0)
        pso = PSO(
            num_particles=10, dim=3,
            bounds=[(-5.0, 5.0)] * 3,
            fitness_func=sphere,
            w=0.5, c1=1.5, c2=1.5,
            max_iters=20,
        )
        result = pso.optimize()
        assert len(result) == 3  # position, fitness, history

    def test_best_position_length_matches_dim(self):
        random.seed(0)
        dim = 6
        pso = PSO(
            num_particles=10, dim=dim,
            bounds=[(-5.0, 5.0)] * dim,
            fitness_func=sphere,
            w=0.5, c1=1.5, c2=1.5,
            max_iters=20,
        )
        best_pos, _, _ = pso.optimize()
        assert len(best_pos) == dim

    def test_history_length(self):
        """History should have num_particles * max_iters entries"""
        random.seed(0)
        num_particles = 10
        max_iters     = 15
        pso = PSO(
            num_particles=num_particles, dim=3,
            bounds=[(-5.0, 5.0)] * 3,
            fitness_func=sphere,
            w=0.5, c1=1.5, c2=1.5,
            max_iters=max_iters,
        )
        _, _, history = pso.optimize()
        assert len(history) == num_particles * max_iters


class TestPSOReproducibility:
    """Tests for PSO seed"""
    def test_same_seed_same_result(self):
        kwargs = dict(
            num_particles=20, dim=5,
            bounds=[(-5.12, 5.12)] * 5,
            fitness_func=sphere,
            w=0.6, c1=1.5, c2=1.5,
            max_iters=30,
        )
        random.seed(123)
        _, fit1, _ = PSO(**kwargs).optimize()

        random.seed(123)
        _, fit2, _ = PSO(**kwargs).optimize()

        assert fit1 == fit2

    def test_different_seeds_may_differ(self):
        kwargs = dict(
            num_particles=20, dim=5,
            bounds=[(-5.12, 5.12)] * 5,
            fitness_func=sphere,
            w=0.6, c1=1.5, c2=1.5,
            max_iters=30,
        )
        random.seed(1)
        _, fit1, _ = PSO(**kwargs).optimize()

        random.seed(9999)
        _, fit2, _ = PSO(**kwargs).optimize()

        assert fit1 != fit2


class TestPSOBounds:
    """Tests for PSO space limits"""
    def test_best_position_within_bounds(self):
        random.seed(42)
        bounds = [(-2.0, 2.0)] * 4
        pso = PSO(
            num_particles=15, dim=4,
            bounds=bounds,
            fitness_func=sphere,
            w=0.5, c1=1.5, c2=1.5,
            max_iters=30,
        )
        best_pos, _, _ = pso.optimize()
        for i, pos in enumerate(best_pos):
            assert bounds[i][0] <= pos <= bounds[i][1], (
                f"dim {i}: {pos} out of bounds {bounds[i]}"
            )