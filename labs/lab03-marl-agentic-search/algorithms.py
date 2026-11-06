"""Небольшие реализации дискретных cooperative MARL-алгоритмов на PyTorch.

Методы: Independent REINFORCE, Independent A2C, Independent PPO (IPPO),
Independent Q-Learning (IQL), VDN, QMIX, MAPPO и COMA. Действия выбираются
каждым агентом только по его локальному наблюдению. QMIX, MAPPO и COMA
используют объединённые наблюдения при обучении; VDN использует общую награду.

Один вызов update принимает один полный эпизод, собранный текущей политикой:
obs, next_obs: [T, N, D]; mask, next_mask: [T, N, A] (bool);
actions: [T, N] (int); rewards, dones: [T]. Общая награда скалярна.
Для on-policy методов вызывайте update сразу после сбора, до других обновлений.
На терминальном переходе значение next_obs игнорируется. Маска содержит хотя
бы одно допустимое действие в нетерминальном состоянии.
"""
from __future__ import annotations

import copy
import random

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


def _mlp(n_in: int, n_out: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(), nn.Linear(hidden, n_out))


def _safe_mask(mask: torch.Tensor) -> torch.Tensor:
    """Допускаем пустую маску только как терминальный заполнитель."""
    mask = mask.bool().clone()
    empty = ~mask.any(dim=-1)
    if empty.any():
        mask[..., 0] |= empty
    return mask


def _masked(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return values.masked_fill(~_safe_mask(mask), -1e9)


class Mixer(nn.Module):
    """QMIX: веса миксера неотрицательны, поэтому Q_tot монотонна по Q_i."""

    def __init__(self, n_agents: int, state_dim: int, hidden: int):
        super().__init__()
        self.n_agents, self.hidden = n_agents, hidden
        self.w1 = nn.Linear(state_dim, n_agents * hidden)
        self.b1 = nn.Linear(state_dim, hidden)
        self.w2 = nn.Linear(state_dim, hidden)
        self.b2 = _mlp(state_dim, 1, hidden)

    def forward(self, q: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        w1 = self.w1(state).abs().view(-1, self.n_agents, self.hidden)
        w2 = self.w2(state).abs().view(-1, self.hidden, 1)
        h = torch.nn.functional.elu(torch.bmm(q.unsqueeze(1), w1) + self.b1(state).unsqueeze(1))
        return (torch.bmm(h, w2).view(-1) + self.b2(state).view(-1))


class MARL:
    """API: MARL(method, n_agents, obs_dim, n_actions, seed, ...).

    act(obs, mask, explore=True) -> ndarray[N]. update(batch) -> dict.
    Поддерживаемые имена: reinforce, ia2c, ippo, iql, vdn, qmix, mappo, coma.
    Для Q-методов explore=True означает epsilon-greedy, иначе greedy;
    для policy-методов explore=True означает выбор из распределения, иначе argmax.
    """

    METHODS = {"reinforce", "ia2c", "ippo", "iql", "vdn", "qmix", "mappo", "coma"}
    Q_METHODS = {"iql", "vdn", "qmix"}

    def __init__(self, method: str, n_agents: int, obs_dim: int, n_actions: int,
                 seed: int = 0, hidden: int = 64, lr: float = 3e-4,
                 gamma: float = 0.99, epsilon: float = 0.10,
                 ppo_epochs: int = 4, ppo_clip: float = 0.2,
                 target_interval: int = 25, entropy_coef: float = 0.01):
        method = method.lower()
        if method not in self.METHODS:
            raise ValueError(f"Unknown method: {method}")
        if min(n_agents, obs_dim, n_actions, hidden, target_interval, ppo_epochs) < 1:
            raise ValueError("Dimensions and intervals must be positive")
        self.method, self.n_agents, self.obs_dim, self.n_actions = method, n_agents, obs_dim, n_actions
        self.gamma, self.epsilon, self.ppo_epochs, self.ppo_clip = gamma, epsilon, ppo_epochs, ppo_clip
        self.target_interval, self.entropy_coef, self.updates = target_interval, entropy_coef, 0
        self.rng = np.random.default_rng(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        self.actors = nn.ModuleList([_mlp(obs_dim, n_actions, hidden) for _ in range(n_agents)])
        self.modules = nn.ModuleList([self.actors])
        state_dim = n_agents * obs_dim
        if method in {"ia2c", "ippo"}:
            self.values = nn.ModuleList([_mlp(obs_dim, 1, hidden) for _ in range(n_agents)])
            self.modules.append(self.values)
        elif method == "mappo":
            self.values = _mlp(state_dim, 1, hidden)
            self.modules.append(self.values)
        elif method == "coma":
            # Для агента i вход содержит state, идентификатор i и действия всех
            # остальных агентов; выход перечисляет Q для всех действий агента i.
            self.critic = _mlp(state_dim + n_agents * n_actions + n_agents, n_actions, hidden)
            self.modules.append(self.critic)
            self.target_critic = copy.deepcopy(self.critic)
        if method == "qmix":
            self.mixer = Mixer(n_agents, state_dim, hidden)
            self.modules.append(self.mixer)
            self.target_mixer = copy.deepcopy(self.mixer)
        if method in self.Q_METHODS:
            self.target_actors = copy.deepcopy(self.actors)
        self.optimizer = torch.optim.Adam(self.modules.parameters(), lr=lr)

    def _outputs(self, obs: torch.Tensor, target: bool = False) -> torch.Tensor:
        nets = self.target_actors if target else self.actors
        return torch.stack([nets[i](obs[:, i]) for i in range(self.n_agents)], dim=1)

    def act(self, obs, mask, explore: bool = True) -> np.ndarray:
        o = torch.as_tensor(np.asarray(obs), dtype=torch.float32).unsqueeze(0)
        m = torch.as_tensor(np.asarray(mask), dtype=torch.bool).unsqueeze(0)
        if o.shape != (1, self.n_agents, self.obs_dim) or m.shape != (1, self.n_agents, self.n_actions):
            raise ValueError("Invalid observation or action-mask shape")
        if not bool(m.any(dim=-1).all()):
            raise ValueError("Each agent needs a legal action when acting")
        with torch.no_grad():
            logits = _masked(self._outputs(o), m)[0]
            if self.method in self.Q_METHODS:
                a = logits.argmax(dim=-1).numpy()
                if explore:
                    for i in range(self.n_agents):
                        if self.rng.random() < self.epsilon:
                            a[i] = self.rng.choice(np.flatnonzero(m[0, i].numpy()))
                return a
            return (Categorical(logits=logits).sample() if explore else logits.argmax(dim=-1)).numpy()

    def _batch(self, batch):
        def tensor(k, dtype):
            return torch.as_tensor(np.asarray(batch[k]), dtype=dtype)
        obs, nxt = tensor("obs", torch.float32), tensor("next_obs", torch.float32)
        mask, nm = tensor("mask", torch.bool), tensor("next_mask", torch.bool)
        actions, rewards, dones = tensor("actions", torch.long), tensor("rewards", torch.float32), tensor("dones", torch.bool)
        t = len(rewards)
        if t < 1 or obs.shape != (t, self.n_agents, self.obs_dim) or nxt.shape != obs.shape:
            raise ValueError("Invalid episode observations")
        if mask.shape != (t, self.n_agents, self.n_actions) or nm.shape != mask.shape or actions.shape != (t, self.n_agents) or dones.shape != (t,):
            raise ValueError("Invalid episode action/reward/mask shapes")
        if not bool(dones[-1]) or bool(dones[:-1].any()):
            raise ValueError("update requires one complete episode with terminal last transition")
        if not bool(mask.any(dim=-1).all()) or not bool(nm[~dones].any(dim=-1).all()):
            raise ValueError("Nonterminal action masks must be nonempty")
        if bool(((actions < 0) | (actions >= self.n_actions)).any()) or not bool(mask.gather(-1, actions.unsqueeze(-1)).all()):
            raise ValueError("Episode includes invalid actions")
        return obs, mask, actions, rewards, nxt, nm, dones.float()

    def _returns(self, rewards, dones):
        ret = torch.zeros_like(rewards)
        value = torch.tensor(0.0)
        for t in range(len(rewards) - 1, -1, -1):
            value = rewards[t] + self.gamma * (1.0 - dones[t]) * value
            ret[t] = value
        return ret

    def _coma_q(self, critic, obs, actions):
        t = obs.shape[0]
        state = obs.reshape(t, -1)
        onehot = torch.nn.functional.one_hot(actions, self.n_actions).float()
        qs = []
        for i in range(self.n_agents):
            other = onehot.clone()
            other[:, i] = 0
            identity = torch.zeros(t, self.n_agents)
            identity[:, i] = 1
            qs.append(critic(torch.cat([state, other.reshape(t, -1), identity], dim=-1)))
        return torch.stack(qs, dim=1)

    def update(self, batch) -> dict:
        obs, mask, actions, reward, nxt, nm, done = self._batch(batch)
        state, next_state = obs.reshape(len(reward), -1), nxt.reshape(len(reward), -1)
        if self.method in self.Q_METHODS:
            q = self._outputs(obs).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            with torch.no_grad():
                nq = _masked(self._outputs(nxt, target=True), nm).max(dim=-1).values
                if self.method == "iql":
                    target = reward[:, None] + self.gamma * (1 - done[:, None]) * nq
                elif self.method == "vdn":
                    target = reward + self.gamma * (1 - done) * nq.sum(dim=-1)
                else:
                    target = reward + self.gamma * (1 - done) * self.target_mixer(nq, next_state)
            pred = q if self.method == "iql" else (q.sum(dim=-1) if self.method == "vdn" else self.mixer(q, state))
            loss = torch.nn.functional.mse_loss(pred, target)
            self.optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(self.modules.parameters(), 10.0); self.optimizer.step()
            self.updates += 1
            if self.updates % self.target_interval == 0:
                self.target_actors.load_state_dict(self.actors.state_dict())
                if self.method == "qmix":
                    self.target_mixer.load_state_dict(self.mixer.state_dict())
            return {"loss": float(loss.detach())}

        with torch.no_grad():
            old_dist = Categorical(logits=_masked(self._outputs(obs), mask))
            old_log = old_dist.log_prob(actions)
        if self.method == "coma":
            dist = Categorical(logits=_masked(self._outputs(obs), mask))
            with torch.no_grad():
                next_dist = Categorical(logits=_masked(self._outputs(nxt), nm))
                next_actions = next_dist.sample()
                next_q = self._coma_q(self.target_critic, nxt, next_actions)
                next_taken = next_q.gather(-1, next_actions.unsqueeze(-1)).squeeze(-1)
                target = reward[:, None] + self.gamma * (1 - done[:, None]) * next_taken
            all_q = self._coma_q(self.critic, obs, actions)
            taken = all_q.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            with torch.no_grad():
                baseline = (dist.probs * all_q).sum(dim=-1)
                advantage = taken - baseline
            loss = -(dist.log_prob(actions) * advantage).mean() + 0.5 * torch.nn.functional.mse_loss(taken, target) - self.entropy_coef * dist.entropy().mean()
            self.optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(self.modules.parameters(), 10.0); self.optimizer.step()
            self.updates += 1
            if self.updates % self.target_interval == 0:
                self.target_critic.load_state_dict(self.critic.state_dict())
            return {"loss": float(loss.detach())}

        returns = self._returns(reward, done)
        n_epochs = self.ppo_epochs if self.method in {"ippo", "mappo"} else 1
        if self.method in {"ippo", "mappo"}:
            with torch.no_grad():
                if self.method == "ippo":
                    old_v = torch.cat([self.values[i](obs[:, i]) for i in range(self.n_agents)], dim=-1)
                else:
                    old_v = self.values(state).expand(-1, self.n_agents)
                fixed_adv = returns[:, None] - old_v
        for _ in range(n_epochs):
            dist = Categorical(logits=_masked(self._outputs(obs), mask))
            logp = dist.log_prob(actions)
            if self.method == "reinforce":
                actor_loss = -(logp * returns[:, None]).mean()
                value_loss = torch.tensor(0.0)
            else:
                if self.method in {"ia2c", "ippo"}:
                    values = torch.cat([self.values[i](obs[:, i]) for i in range(self.n_agents)], dim=-1)
                else:
                    values = self.values(state).expand(-1, self.n_agents)
                adv = fixed_adv if self.method in {"ippo", "mappo"} else (returns[:, None] - values).detach()
                if self.method in {"ippo", "mappo"}:
                    ratio = (logp - old_log).exp()
                    actor_loss = -torch.minimum(ratio * adv, ratio.clamp(1 - self.ppo_clip, 1 + self.ppo_clip) * adv).mean()
                else:
                    actor_loss = -(logp * adv).mean()
                value_loss = (values - returns[:, None]).square().mean()
            loss = actor_loss + 0.5 * value_loss - self.entropy_coef * dist.entropy().mean()
            self.optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(self.modules.parameters(), 10.0); self.optimizer.step()
        self.updates += 1
        return {"loss": float(loss.detach()), "actor_loss": float(actor_loss.detach()), "value_loss": float(value_loss.detach())}
