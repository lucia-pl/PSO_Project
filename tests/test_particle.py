"""
@file        test_particle.py
@author      Lucía Parreño Legorburo
@brief       Tests for src/core/particle.py
"""

import random
from src.core.particle import Particle


class TestParticleInit:
    """Tests for particle initialization"""
    def test_position_within_bounds(self, simple_bounds):
        random.seed(42)
        p = Particle(dim=10, bounds=simple_bounds)
        for i, pos in enumerate(p.position):
            assert simple_bounds[i][0] <= pos <= simple_bounds[i][1]

    def test_velocity_length_matches_dim(self, simple_bounds):
        random.seed(42)
        p = Particle(dim=10, bounds=simple_bounds)
        assert len(p.velocity) == 10

    def test_initial_best_fitness_is_inf(self, simple_bounds):
        p = Particle(dim=10, bounds=simple_bounds)
        assert p.best_fitness == float("inf")

    def test_reproducibility_with_seed(self, simple_bounds):
        random.seed(99)
        p1 = Particle(dim=10, bounds=simple_bounds)
        random.seed(99)
        p2 = Particle(dim=10, bounds=simple_bounds)
        assert p1.position == p2.position
        assert p1.velocity == p2.velocity


class TestParticleEvaluate:
    """Tests for particle evaluation"""
    def test_evaluate_updates_best_fitness(self, simple_bounds):
        random.seed(42)
        p = Particle(dim=10, bounds=simple_bounds)
        fitness = p.evaluate_particle(lambda x: sum(xi**2 for xi in x))
        assert p.best_fitness == fitness

    def test_best_fitness_never_worsens(self, simple_bounds):
        random.seed(42)
        p = Particle(dim=10, bounds=simple_bounds)

        best_so_far = float("inf")
        for _ in range(20):
            p.position = [random.uniform(b[0], b[1]) for b in simple_bounds]
            p.evaluate_particle(lambda x: sum(xi**2 for xi in x))
            assert p.best_fitness <= best_so_far
            best_so_far = p.best_fitness

    def test_best_position_updated_on_improvement(self, simple_bounds):
        random.seed(42)
        p = Particle(dim=10, bounds=simple_bounds)
        p.evaluate_particle(lambda x: sum(xi**2 for xi in x))
        assert p.best_position == p.position


class TestParticleUpdate:
    """Test for particle updates"""
    def test_position_stays_within_bounds_after_update(self, simple_bounds):
        random.seed(42)
        p = Particle(dim=10, bounds=simple_bounds)
        global_best = [0.0] * 10

        p.evaluate_particle(lambda x: sum(xi**2 for xi in x))

        for _ in range(50):
            p.update_velocity(global_best=global_best, w=0.5, c1=1.5, c2=1.5)
            p.update_position(simple_bounds)
            for i, pos in enumerate(p.position):
                assert simple_bounds[i][0] <= pos <= simple_bounds[i][1], (
                    f"dim {i}: {pos} out of bounds {simple_bounds[i]}"
                )