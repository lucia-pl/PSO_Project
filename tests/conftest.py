"""
@file        conftest.py
@author      Lucía Parreño Legorburo
@brief       Configurations of tests with pytest
"""
import pytest
import random


@pytest.fixture
def fixed_seed():
    """Sets a fixed seed before each test for reproducibility."""
    random.seed(42)
    return 42


@pytest.fixture
def simple_bounds():
    return [(-5.12, 5.12)] * 10


@pytest.fixture
def simple_bounds_2d():
    return [(-5.12, 5.12)] * 2