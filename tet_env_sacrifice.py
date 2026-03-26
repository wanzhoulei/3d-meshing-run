"""
SacrificeVecEnv — vectorized tet mesh refinement environment with a
lookback reward designed to allow the agent to learn sacrifice moves.

Motivation
----------
In the base TetMeshRefineVecEnv the reward is:
    r_t = (score_t - score_{t-1}) * reward_scale

This reward directly *punishes* any move that temporarily lowers quality,
which prevents the agent from ever learning sacrifice moves (temporarily
worsening quality to unlock a later improvement).

Fix: replace the 1-step delta with a w-step lookback:
    r_t = (score_t - score_{t-w}) * reward_scale

Short-term sacrifices (quality dips lasting < w steps) now yield near-zero
reward instead of negative reward, while multi-step improvements accumulate
a positive signal.  The termination bookkeeping (patience, best_score) still
tracks the single-step quality so that episode length is not artificially
inflated.

Usage
-----
    from tet_env_sacrifice import SacrificeVecEnv

    env = SacrificeVecEnv(
        make_mesh_fn=make_mesh,
        num_envs=32,
        lookback_window=8,        # ← new parameter; all others same as base
        max_steps_per_episode=150,
        patience=50,
        reward_scale=10,
        tet_quality_mode="mean_ratio",
    )
"""
from __future__ import annotations

from collections import deque
from typing import Callable, Optional

import numpy as np

from tet_env import TetMeshRefineVecEnv, StepResult
from tet_mesh_topology_local import TetMeshTopology


class SacrificeVecEnv(TetMeshRefineVecEnv):
    """
    Subclass of TetMeshRefineVecEnv that replaces the per-step reward with a
    lookback-window reward to facilitate learning sacrifice moves.

    All constructor arguments are identical to TetMeshRefineVecEnv plus one
    new keyword argument:

    Parameters
    ----------
    lookback_window : int
        Width w of the reward lookback.  r_t = (q_t - q_{t-w}) * reward_scale.
        Must be >= 1.  w=1 recovers the base-class per-step reward.
    """

    def __init__(
        self,
        make_mesh_fn: Callable[[], tuple[np.ndarray, np.ndarray]],
        num_envs: int = 8,
        seed: Optional[int] = None,
        max_steps_per_episode: int = 256,
        eps_improve: float = 1e-6,
        patience: int = 50,
        invalid_penalty: float = 0.0,
        reward_scale: float = 1.0,
        tet_quality_mode: str = "mean_ratio",
        score_mode: str = "softmin",
        softmin_tau: float = 0.05,
        worstk_k: int = 10,
        lookback_window: int = 8,
    ):
        # Store before calling super().__init__() because super().__init__()
        # calls self.reset(), which needs self.lookback_window.
        self.lookback_window: int = max(1, int(lookback_window))
        # Placeholder; populated in reset().
        self._quality_history: list[deque] = []

        super().__init__(
            make_mesh_fn=make_mesh_fn,
            num_envs=num_envs,
            seed=seed,
            max_steps_per_episode=max_steps_per_episode,
            eps_improve=eps_improve,
            patience=patience,
            invalid_penalty=invalid_penalty,
            reward_scale=reward_scale,
            tet_quality_mode=tet_quality_mode,
            score_mode=score_mode,
            softmin_tau=softmin_tau,
            worstk_k=worstk_k,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_env_history(self, b: int) -> None:
        """
        Initialise the quality-history deque for env b by filling it with the
        current score.  Called at episode start / reset.
        """
        sc = self._score(self.topos[b])
        self._quality_history[b] = deque(
            [sc] * self.lookback_window, maxlen=self.lookback_window
        )

    # ------------------------------------------------------------------
    # Overridden public API
    # ------------------------------------------------------------------

    def reset(self) -> list[dict]:
        """
        Reset all environments and initialise per-env quality-history deques.
        """
        obs = super().reset()
        # Allocate or re-allocate history list after base reset has rebuilt topos.
        self._quality_history = [None] * self.num_envs  # type: ignore[list-item]
        for b in range(self.num_envs):
            self._init_env_history(b)
        return obs

    def step(self, actions: np.ndarray) -> StepResult:
        """
        Step all environments in parallel.

        Reward is the w-step lookback quality delta:
            r_t = (score_t - score_{t-w}) * reward_scale

        All other semantics (done conditions, obs, info) are identical to
        the base class.
        """
        assert actions.shape == (self.num_envs,)

        rewards = np.zeros((self.num_envs,), dtype=np.float64)
        dones = np.zeros((self.num_envs,), dtype=bool)
        infos: list[dict] = []
        obs: list[dict] = []

        for b in range(self.num_envs):
            topo = self.topos[b]
            a = int(actions[b])

            score_before = self._score(topo)
            valid = topo.apply_action(a)

            if not valid:
                rewards[b] = -self.invalid_penalty
                score_now = score_before
            else:
                score_now = self._score(topo)
                # Lookback reward: compare current quality to w steps ago.
                score_w_ago = self._quality_history[b][0]  # oldest entry
                rewards[b] = (score_now - score_w_ago) * self.reward_scale

            # Push current quality into the rolling window.
            self._quality_history[b].append(score_now)

            # Termination bookkeeping (unchanged from base class).
            self.steps[b] += 1
            improved = (score_now - self.best_score[b]) >= self.eps_improve
            if improved:
                self.best_score[b] = score_now
                self.no_improve[b] = 0
            else:
                self.no_improve[b] += 1

            candidate_mask = topo.candidate_mask()
            no_moves = not bool(np.any(candidate_mask))
            too_long = self.steps[b] >= self.max_steps_per_episode
            stalled = self.no_improve[b] >= self.patience
            done = no_moves or too_long or stalled
            dones[b] = done

            infos.append(dict(
                episode_id=int(self.episode_id[b]),
                step=int(self.steps[b]),
                score=float(score_now),
                best_score=float(self.best_score[b]),
                no_improve_steps=int(self.no_improve[b]),
                action=int(a),
                action_valid=bool(valid),
                no_moves=bool(no_moves),
                too_long=bool(too_long),
                stalled=bool(stalled),
            ))

            if done:
                # Auto-reset this environment.
                P, T = self.make_mesh_fn()
                self.topos[b] = TetMeshTopology(
                    P, T, tet_quality_mode=self.tet_quality_mode
                )
                self.steps[b] = 0
                self.episode_id[b] += 1
                sc = self._score(self.topos[b])
                self.best_score[b] = sc
                self.no_improve[b] = 0
                self._init_env_history(b)
                obs.append(self._obs(self.topos[b], b))
            else:
                obs.append(self._obs(topo, b, candidate_mask=candidate_mask))

        return StepResult(obs=obs, reward=rewards, done=dones, info=infos)
