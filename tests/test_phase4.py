"""
Phase 4 unit tests — Baseline robust aggregation.

Tests verify:
    1. Median aggregation correctness.
    2. Trimmed Mean aggregation correctness (including trimming out extremes).
    3. Krum selection correctness (picks the geometric median-like point).
    4. Multi-Krum selection correctness.

Run with:
    cd BTP-FL-Poisoning
    .venv/bin/python -m pytest tests/test_phase4.py -v
"""

import os
import sys

import pytest
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from proposed.aggregation.baselines import median, trimmed_mean, krum, multi_krum


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_updates():
    """
    Returns 5 updates (all same size).
    Updates 0, 1, 2 are similar.
    Update 3 is an extreme outlier (positive).
    Update 4 is an extreme outlier (negative).
    """
    # Create simple 1-parameter state_dicts
    def make_sd(val: float):
        return {"layer1": torch.tensor([val, val], dtype=torch.float32)}

    updates = [
        (make_sd(1.0), 10),
        (make_sd(1.1), 10),
        (make_sd(0.9), 10),
        (make_sd(100.0), 10),  # Poison 1
        (make_sd(-100.0), 10), # Poison 2
    ]
    return updates


# ── 1. Median ──────────────────────────────────────────────────────────────────

class TestMedian:
    def test_median_ignores_outliers(self, mock_updates):
        agg = median(mock_updates)
        # Median of [-100.0, 0.9, 1.0, 1.1, 100.0] is 1.0
        assert torch.allclose(agg["layer1"], torch.tensor([1.0, 1.0]))

    def test_median_empty_raises(self):
        with pytest.raises(ValueError):
            median([])


# ── 2. Trimmed Mean ────────────────────────────────────────────────────────────

class TestTrimmedMean:
    def test_trimmed_mean_removes_outliers(self, mock_updates):
        # 5 clients. trim_ratio=0.2 means trim top 1 (0.2*5) and bottom 1 (0.2*5)
        # Surviving: [0.9, 1.0, 1.1]. Mean = 1.0
        agg = trimmed_mean(mock_updates, trim_ratio=0.2)
        assert torch.allclose(agg["layer1"], torch.tensor([1.0, 1.0]))

    def test_trimmed_mean_no_trim(self, mock_updates):
        # trim_ratio=0.0 means normal mean
        agg = trimmed_mean(mock_updates, trim_ratio=0.0)
        # sum = 2.0 + 100.0 - 100.0 + 1.0 = 3.0 / 5 = 0.6
        assert torch.allclose(agg["layer1"], torch.tensor([0.6, 0.6]))

    def test_trimmed_mean_too_high_ratio(self, mock_updates):
        # trim 0.6 * 5 = 3. Top 3 and bottom 3 trimmed -> 6 trimmed out of 5!
        with pytest.raises(ValueError):
            trimmed_mean(mock_updates, trim_ratio=0.6)


# ── 3. Krum ────────────────────────────────────────────────────────────────────

class TestKrum:
    def test_krum_selects_benign(self, mock_updates):
        # We have 5 clients, f=2 assumed attackers.
        # N - f - 2 = 5 - 2 - 2 = 1 nearest neighbor.
        # Distances:
        # 0 (1.0) to 1 (1.1) is 0.1
        # 1 (1.1) to 0 (1.0) is 0.1
        # 2 (0.9) to 0 (1.0) is 0.1
        # Outliers have distances ~100
        # So Krum should select 0, 1, or 2 (likely 0 or 1 or 2 as they have same min distance)
        # Actually 1.0 is between 0.9 and 1.1, so its score is min.
        # distances to others:
        # for 1.0: 1.1(0.1^2), 0.9(0.1^2) => sum of 1 smallest = 0.01
        agg = krum(mock_updates, f=2)
        
        # It should pick one of the benign ones
        val = agg["layer1"][0].item()
        assert val in [0.9, 1.0, 1.1]

    def test_krum_too_many_attackers(self, mock_updates):
        # N=5. N - f - 2 >= 0 => f <= 3
        with pytest.raises(ValueError):
            krum(mock_updates, f=4)


# ── 4. Multi-Krum ──────────────────────────────────────────────────────────────

class TestMultiKrum:
    def test_multi_krum_averages_top_m(self, mock_updates):
        # Pick top m=3. Should pick the three benign ones and average them.
        # Benign ones: 0.9, 1.0, 1.1. Mean = 1.0
        agg = multi_krum(mock_updates, f=2, m=3)
        assert torch.allclose(agg["layer1"], torch.tensor([1.0, 1.0]))

    def test_multi_krum_m_too_large(self, mock_updates):
        with pytest.raises(ValueError):
            multi_krum(mock_updates, f=1, m=6)
