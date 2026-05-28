"""
@file        test_swarm.py
@author      Lucía Parreño Legorburo
@brief       Tests for src/core/swarm.py
"""

import random
from src.core.swarm import Swarm
from src.objectives.benchmarks import sphere


class TestSwarmInit:
    """Tests for the swarm initialization"""
    def test_correct_number_of_particles(self, simple_bounds):
        random.seed(42)
        s = Swarm(num_particles=20, dim=10, bounds=simple_bounds)
        assert len(s.particles) == 20

    def test_global_best_fitness_starts_inf(self, simple_bounds):
        s = Swarm(num_particles=10, dim=10, bounds=simple_bounds)
        assert s.global_best_fitness == float("inf")

    def test_global_best_history_starts_empty(self, simple_bounds):
        s = Swarm(num_particles=10, dim=10, bounds=simple_bounds)
        assert s.global_best_history == []


class TestSwarmEvaluate:
    """Tests for particles in the swarm evaluation"""
    def test_global_best_fitness_updated_after_evaluate(self, simple_bounds):
        random.seed(42)
        s = Swarm(num_particles=10, dim=10, bounds=simple_bounds)
        s.evaluate(sphere)
        assert s.global_best_fitness < float("inf")

    def test_global_best_fitness_monotonically_non_increasing(self, simple_bounds):
        random.seed(42)
        s = Swarm(num_particles=20, dim=10, bounds=simple_bounds)
        prev_best = float("inf")

        for _ in range(30):
            s.evaluate(sphere)
            s.update(bounds=simple_bounds, w=0.5, c1=1.5, c2=1.5)
            assert s.global_best_fitness <= prev_best
            prev_best = s.global_best_fitness

    def test_history_grows_after_each_evaluate(self, simple_bounds):
        random.seed(42)
        s = Swarm(num_particles=10, dim=10, bounds=simple_bounds)
        s.evaluate(sphere)
        assert len(s.global_best_history) == 10  


class TestSwarmReproducibility:
    """Tests addaptability of the swarm with the seeds"""
    def test_same_seed_same_result(self, simple_bounds):
        random.seed(7)
        s1 = Swarm(num_particles=15, dim=10, bounds=simple_bounds)
        for _ in range(10):
            s1.evaluate(sphere)
            s1.update(bounds=simple_bounds, w=0.7, c1=1.5, c2=1.5)

        random.seed(7)
        s2 = Swarm(num_particles=15, dim=10, bounds=simple_bounds)
        for _ in range(10):
            s2.evaluate(sphere)
            s2.update(bounds=simple_bounds, w=0.7, c1=1.5, c2=1.5)

        assert s1.global_best_fitness == s2.global_best_fitness
        assert s1.global_best_position == s2.global_best_position