"""Tests for ``mars_rec.torch.optimizer.RAdam``.

The custom RAdam implements rectified-variance (Liu et al., 2019). These
tests verify the public contract:
- ``step()`` updates parameters via gradient descent.
- The optimizer keeps per-parameter state (``step``, ``exp_avg``,
  ``exp_avg_sq``) and increments ``step`` monotonically.
- Param groups are preserved.
- For the same input, RAdam produces a different trajectory than vanilla
  Adam (rectification kicks in vs. Adam's plain bias correction).
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn

from mars_rec.torch.optimizer import RAdam


# The implementation uses deprecated positional-value addcmul_/add_ APIs.
# Silence them at the test boundary so the suite stays green on newer torch.
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestRAdamConstruction:
    def test_invalid_lr_raises(self):
        m = nn.Linear(4, 1)
        with pytest.raises(ValueError, match="learning rate"):
            RAdam(m.parameters(), lr=-1.0)

    def test_invalid_eps_raises(self):
        m = nn.Linear(4, 1)
        with pytest.raises(ValueError, match="epsilon"):
            RAdam(m.parameters(), eps=-1e-8)

    def test_invalid_beta_raises(self):
        m = nn.Linear(4, 1)
        with pytest.raises(ValueError, match="beta"):
            RAdam(m.parameters(), betas=(1.5, 0.999))

    def test_two_param_groups_preserved(self):
        m1 = nn.Linear(4, 2)
        m2 = nn.Linear(2, 1)
        opt = RAdam(
            [
                {"params": m1.parameters(), "lr": 1e-2},
                {"params": m2.parameters(), "lr": 1e-3},
            ]
        )
        assert len(opt.param_groups) == 2
        assert opt.param_groups[0]["lr"] == 1e-2
        assert opt.param_groups[1]["lr"] == 1e-3


# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------


class TestRAdamStep:
    def test_step_updates_parameters(self):
        torch.manual_seed(0)
        m = nn.Linear(4, 1)
        before = m.weight.detach().clone()

        opt = RAdam(m.parameters(), lr=1e-2)
        # Force a non-trivial gradient.
        x = torch.randn(8, 4)
        y = torch.randn(8, 1)
        loss = ((m(x) - y) ** 2).mean()
        loss.backward()
        opt.step()

        after = m.weight.detach()
        # The "degenerated to SGD" branch always moves the params in the
        # first few steps, before the variance-rectification term kicks in.
        assert not torch.equal(before, after)

    def test_state_step_increments_per_call(self):
        torch.manual_seed(0)
        m = nn.Linear(4, 1)
        opt = RAdam(m.parameters(), lr=1e-2)

        for expected in range(1, 6):
            opt.zero_grad()
            loss = m(torch.randn(2, 4)).sum()
            loss.backward()
            opt.step()
            assert opt.state[m.weight]["step"] == expected

    def test_state_contains_expected_keys(self):
        m = nn.Linear(4, 1)
        opt = RAdam(m.parameters(), lr=1e-2)
        loss = m(torch.randn(2, 4)).sum()
        loss.backward()
        opt.step()
        state = opt.state[m.weight]
        assert "step" in state
        assert "exp_avg" in state and state["exp_avg"].shape == m.weight.shape
        assert "exp_avg_sq" in state and state["exp_avg_sq"].shape == m.weight.shape

    def test_radam_diverges_from_adam_after_few_steps(self):
        """Same seed, same init, same data → RAdam and Adam must produce
        different parameter trajectories (the rectified variance branch is
        the whole point of RAdam)."""
        torch.manual_seed(1234)
        m_radam = nn.Linear(4, 1)
        torch.manual_seed(1234)
        m_adam = nn.Linear(4, 1)

        # Sanity: identical init.
        assert torch.equal(m_radam.weight, m_adam.weight)

        opt_radam = RAdam(m_radam.parameters(), lr=1e-2)
        opt_adam = torch.optim.Adam(m_adam.parameters(), lr=1e-2)

        torch.manual_seed(99)
        xs = [torch.randn(4, 4) for _ in range(8)]
        ys = [torch.randn(4, 1) for _ in range(8)]
        for x, y in zip(xs, ys):
            opt_radam.zero_grad()
            ((m_radam(x) - y) ** 2).mean().backward()
            opt_radam.step()

            opt_adam.zero_grad()
            ((m_adam(x) - y) ** 2).mean().backward()
            opt_adam.step()

        assert not torch.allclose(m_radam.weight, m_adam.weight)
