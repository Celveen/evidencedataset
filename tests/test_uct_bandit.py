"""Tests for UCT math (modality novelty) and the Thompson bandit."""

import pytest

from evidencetree.actions import AnswerAction, ImageSearchAction, TextSearchAction
from evidencetree.mcts import ThompsonBandit, modality_novelty, uct_score
from evidencetree.mcts.uct import UNVISITED_BASE


# --------------------------------------------------------------------------- #
# Modality novelty
# --------------------------------------------------------------------------- #
def test_novelty_rewards_new_action_type():
    path = [TextSearchAction(query="a")]
    assert modality_novelty(ImageSearchAction(), path) == pytest.approx(1.5)  # new type + new granularity
    assert modality_novelty(TextSearchAction(query="b"), path) == pytest.approx(0.0)


def test_novelty_rewards_new_granularity_only_once():
    path = [TextSearchAction(query="a"), ImageSearchAction()]  # whole used
    region = ImageSearchAction(region=(0.1, 0.1, 0.5, 0.5))
    assert modality_novelty(region, path) == pytest.approx(0.5)  # type seen, region new
    path_with_region = path + [region]
    assert modality_novelty(region, path_with_region) == pytest.approx(0.0)


def test_novelty_zero_for_answer():
    assert modality_novelty(AnswerAction(text="x"), []) == 0.0


def test_novelty_custom_weights():
    nu = modality_novelty(ImageSearchAction(), [TextSearchAction(query="a")],
                          w_mod=2.0, w_gran=0.25)
    assert nu == pytest.approx(2.25)


# --------------------------------------------------------------------------- #
# UCT score
# --------------------------------------------------------------------------- #
def test_uct_unvisited_dominates():
    visited = uct_score(q_value=1.0, visits=5, parent_visits=10)
    unvisited = uct_score(q_value=0.0, visits=0, parent_visits=10)
    assert unvisited > visited
    assert unvisited >= UNVISITED_BASE


def test_uct_exploration_decays_with_visits():
    few = uct_score(q_value=0.5, visits=1, parent_visits=20)
    many = uct_score(q_value=0.5, visits=15, parent_visits=20)
    assert few > many


def test_uct_lambda_scales_novelty():
    low = uct_score(q_value=0.5, visits=2, parent_visits=10, novelty=1.5, lam=0.1)
    high = uct_score(q_value=0.5, visits=2, parent_visits=10, novelty=1.5, lam=1.0)
    assert high - low == pytest.approx(0.9 * 1.5)


# --------------------------------------------------------------------------- #
# Thompson bandit (Algorithm 1)
# --------------------------------------------------------------------------- #
def test_warmup_round_robin_covers_three_arms():
    bandit = ThompsonBandit(seed=0)
    picks = []
    for _ in range(3):
        picks.append(bandit.select())
        bandit.update(0.5)
    assert picks == [0.1, 0.5, 1.0]


def test_select_twice_without_update_raises():
    bandit = ThompsonBandit()
    bandit.select()
    with pytest.raises(RuntimeError):
        bandit.select()


def test_update_before_select_raises():
    with pytest.raises(RuntimeError):
        ThompsonBandit().update(0.5)


def test_posterior_learns_good_arm():
    """Feeding high rewards only when arm 1.0 is played should raise its
    posterior mean above the others."""
    bandit = ThompsonBandit(seed=0)
    for _ in range(40):
        lam = bandit.select()
        bandit.update(0.9 if lam == 1.0 else 0.1)
    means = dict(zip(bandit.arms, bandit.posterior_means))
    assert bandit.converged_lambda() == 1.0
    assert means[1.0] > max(v for k, v in means.items() if k != 1.0)


def test_first_reward_only_seeds_history():
    bandit = ThompsonBandit(seed=0)
    bandit.select()
    bandit.update(0.9)  # t=1: no posterior change (Algorithm 1, line 12)
    assert bandit.posterior_means == [0.5] * len(bandit.arms)
    assert bandit.history == [0.9]


def test_bandits_are_independent_per_instance():
    """Per-query independence: separate instances share no state."""
    b1, b2 = ThompsonBandit(seed=0), ThompsonBandit(seed=0)
    for _ in range(10):
        b1.select()
        b1.update(1.0)
    assert b2.posterior_means == [0.5] * len(b2.arms)
    assert b2.history == []


def test_seeded_determinism():
    def run(seed):
        b = ThompsonBandit(seed=seed)
        out = []
        for i in range(8):
            lam = b.select()
            out.append(lam)
            b.update(0.3 + 0.1 * (i % 3))
        return out

    assert run(7) == run(7)
