"""Tests for the UCB1 selection math (no novelty / bandit — both removed)."""

import math

import pytest

from evidencetree.mcts import uct_score
from evidencetree.mcts.uct import UNVISITED_BASE


def test_unvisited_child_sorts_first():
    """An unvisited edge outranks any visited one (classic UCT)."""
    visited = uct_score(q_value=1.0, visits=5, parent_visits=10)
    unvisited = uct_score(q_value=0.0, visits=0, parent_visits=10)
    assert unvisited > visited
    assert unvisited >= UNVISITED_BASE


def test_exploration_decreases_with_visits():
    """More visits -> smaller exploration bonus for the same Q."""
    few = uct_score(q_value=0.5, visits=1, parent_visits=20)
    many = uct_score(q_value=0.5, visits=15, parent_visits=20)
    assert few > many


def test_uct_is_q_plus_ucb1():
    """Score equals Q + c*sqrt(ln N_parent / N_child) exactly."""
    q, c, vis, parent = 0.4, 1.0, 4, 16
    expected = q + c * math.sqrt(math.log(parent) / vis)
    assert uct_score(q_value=q, visits=vis, parent_visits=parent, c=c) == pytest.approx(expected)
