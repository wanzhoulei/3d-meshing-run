from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Dict, Any, Optional

from tet_mesh_topology_local import TetMeshTopology


def softmin_score(q: np.ndarray, tau: float = 0.05) -> float:
    """
    Smooth proxy for min(q): -tau * log(mean(exp(-q/tau))).
    Larger is better, upper bounded by max(q).
    """
    q = np.asarray(q, dtype=np.float64)
    x = np.clip(-q / max(tau, 1e-12), -200.0, 200.0)
    return float(-tau * np.log(np.mean(np.exp(x)) + 1e-18))


def worstk_mean_score(q: np.ndarray, k: int = 10) -> float:
    """Mean of the k smallest qualities (larger is better)."""
    q = np.asarray(q, dtype=np.float64)
    k = int(max(1, min(k, q.size)))
    idx = np.argpartition(q, k - 1)[:k]
    return float(np.mean(q[idx]))


@dataclass
class StepResult:
    """
    This is the data class that stores the results of taking one step
    For a vectorized many envs
    """
    obs: List[Dict[str, Any]]
    reward: np.ndarray
    done: np.ndarray
    info: List[Dict[str, Any]]


class TetMeshRefineVecEnv:
    """
    Vectorized tetra-mesh refinement env (RL-friendly) that mirrors your 2D env structure.

    Each env holds one TetMeshTopology and supports actions:
      - 0..F-1: choose a face -> attempt 2-3
      - F..F+E-1: choose an edge -> attempt 3-2 (only valence==3 edges are eligible)

    Termination (per env):
      - no valid moves remain, OR
      - no improvement in score for 'patience' steps, OR
      - steps reach max_steps_per_episode

    When an env terminates, it is immediately reset with a fresh mesh sample.
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
        score_mode: str = "softmin",   # "softmin" or "worstk"
        softmin_tau: float = 0.05,
        worstk_k: int = 10,
    ):
        """
        The constructor for TetMeshRefineVecEnv object
        Arguments: 
            make_mesh_f: function handler that samples a random data point 
                        It should return a tuple of P and T
            num_envs: int, number of vectorized envs 
            seed: int, random seed 
            max_steps_per_episode: The max steps allowed in each episode 
                        If it reaches this step before termination, the episode will terminate 
            eps_improve: float, the smallest improvement allowed to call "imprvement"
            patience: int, number of steps allowed without improvement before termination 
            invalid_penalty: float, default 0, the penalty applied if invalid step is proposed
            reward_scale: scalar to scale the reward 
            score_mode: str, method to score the current mesh, reward is defined as the difference
            softmin_tau: hyperparameter in the score function, if using softmin score
            worstk_k: hyperparameter in the score function, if using worstk score 
        """
        self.make_mesh_fn = make_mesh_fn #function to draw a random sample of mesh 
        self.num_envs = int(num_envs)
        self.rng = np.random.default_rng(seed)

        self.max_steps_per_episode = int(max_steps_per_episode)
        self.eps_improve = float(eps_improve)
        self.patience = int(patience)
        self.invalid_penalty = float(invalid_penalty)
        self.reward_scale = float(reward_scale)

        self.score_mode = str(score_mode)
        self.softmin_tau = float(softmin_tau)
        self.worstk_k = int(worstk_k)

        self.topos: List[TetMeshTopology] = []
        self.steps = np.zeros((self.num_envs,), dtype=np.int32)
        self.best_score = np.full((self.num_envs,), -np.inf, dtype=np.float64)
        self.no_improve = np.zeros((self.num_envs,), dtype=np.int32)
        self.episode_id = np.zeros((self.num_envs,), dtype=np.int32)

        self.reset()

    def _score(self, topo: TetMeshTopology) -> float:
        """
        Compute and return the score of one particular mesh topology
        """

        if self.score_mode == "softmin":
            return softmin_score(topo.tet_quality, tau=self.softmin_tau)
        elif self.score_mode == "worstk":
            return worstk_mean_score(topo.tet_quality, k=self.worstk_k)
        else:
            raise ValueError(f"Unknown score_mode={self.score_mode}")

    def _obs(
        self,
        topo: TetMeshTopology,
        env_idx: Optional[int] = None,
        candidate_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Returns the observation of one particular topology
        """
        step_frac = 0.0
        no_improve_frac = 0.0
        if env_idx is not None:
            step_frac = float(self.steps[env_idx]) / float(max(1, self.max_steps_per_episode))
            no_improve_frac = float(self.no_improve[env_idx]) / float(max(1, self.patience))
        if candidate_mask is None:
            candidate_mask = topo.candidate_mask()
        return dict(
            points=topo.points,  #shape (N, 3)
            tets=topo.tets, #shape (K, 4)
            faces=topo.faces,#shape (F, 3)
            face2tet=topo.face2tet, #shape shape (F, 2)
            edges=topo.edges, # shape (2, E)
            candidate_mask=candidate_mask, # shape (F+E,)
            tet_quality=topo.tet_quality, # shape (K,)
            step_frac=step_frac,
            no_improve_frac=no_improve_frac,
        )

    def reset(self) -> List[Dict[str, Any]]:
        """
        This function builds a list of env

        It samples num_envs topologies and set all fields
        Returns a list of observation dictionaries for each env
        """

        self.topos = []
        for b in range(self.num_envs):
            P, T = self.make_mesh_fn()
            self.topos.append(TetMeshTopology(P, T))
            self.steps[b] = 0
            sc = self._score(self.topos[b])
            self.best_score[b] = sc
            self.no_improve[b] = 0
            self.episode_id[b] = 0
        return [self._obs(self.topos[b], b) for b in range(self.num_envs)]

    def step(self, actions: np.ndarray) -> StepResult:
        """
        This method takes one step in parallel for all envs 

        Arguments:
            actions: np.ndarray, shape (num_envs,), the action id for each env to take 
        Returns:
            A StepResult object that describes the result of this step it contains:
                obs: List[Dict[str, Any]], the obs dict of each env after the step 
                reward: np.ndarray, the reward in each env
                done: np.ndarray, whether this episode is done for each env 
                info: List[Dict[str, Any]], info about each env:
                    episode_id=int(self.episode_id[b]), current episode number
                    step=int(self.steps[b]), current step number 
                    score=float(score_now), current score
                    best_score=float(self.best_score[b]), best score ever in current episode 
                    no_improve_steps=int(self.no_improve[b]), number of steps with no improvement in current episode
                    action=int(a), id of action taken
                    action_valid=bool(valid), whether the action was valid 
                    no_moves=bool(no_moves), whether the action is not enforced 
                    too_long=bool(too_long), 
                    stalled=bool(stalled),
        """

        assert actions.shape == (self.num_envs,)

        rewards = np.zeros((self.num_envs,), dtype=np.float64)
        dones = np.zeros((self.num_envs,), dtype=bool)
        infos: List[Dict[str, Any]] = []
        obs: List[Dict[str, Any]] = []

        for b in range(self.num_envs):
            topo = self.topos[b]
            a = int(actions[b])

            score_before = self._score(topo)
            valid = topo.apply_action(a)
            if not valid:
                rewards[b] = -self.invalid_penalty
                score_now = score_before
            else:
                score_after = self._score(topo)
                rewards[b] = (score_after - score_before) * self.reward_scale
                score_now = score_after

            # update termination bookkeeping
            self.steps[b] += 1

            improved = (score_now - self.best_score[b]) >= self.eps_improve
            if improved:
                self.best_score[b] = score_now
                self.no_improve[b] = 0
            else:
                self.no_improve[b] += 1

            # done conditions
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
                # auto reset this env
                P, T = self.make_mesh_fn()
                self.topos[b] = TetMeshTopology(P, T)
                self.steps[b] = 0
                self.episode_id[b] += 1
                sc = self._score(self.topos[b])
                self.best_score[b] = sc
                self.no_improve[b] = 0
                obs.append(self._obs(self.topos[b], b))
            else:
                # Reuse candidate_mask already computed above.
                obs.append(self._obs(topo, b, candidate_mask=candidate_mask))

        return StepResult(obs=obs, reward=rewards, done=dones, info=infos)
