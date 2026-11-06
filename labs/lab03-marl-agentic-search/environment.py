"""Cooperative search on precomputed, label-free BM25 results."""
import numpy as np


class SearchEnvironment:
    ACTIONS = 6  # stop; search variant 1..3 or select rank 1..5
    OBS = 21

    def __init__(self, row, judgments, n_agents=2, communication=True, cost=.02, stress=False):
        if n_agents not in (1, 2):
            raise ValueError('Only one or two agents are supported')
        self.row, self.judgments, self.n = row, judgments, n_agents
        self.communication, self.cost, self.stress = communication, cost, stress
        self.rounds = 4 // n_agents
        self.t = 0
        self.variants = [-1] * n_agents
        self.last_actions = [0] * n_agents
        self.selected = []
        self.searches = 0
        self.duplicate_searches = 0
        self.mask_checks = 0
        self.done_agent = [False] * n_agents

    def hits(self, agent):
        variant = self.variants[agent]
        if variant < 0:
            return []
        hits = self.row['variants'][variant]['hits']
        return hits[::2] if self.stress else hits

    def observe(self):
        obs = np.zeros((self.n, self.OBS), dtype=np.float32)
        mask = np.zeros((self.n, self.ACTIONS), dtype=bool)
        for i in range(self.n):
            mask[i, 0] = True
            obs[i, 0] = self.t / max(1, self.rounds * 2 - 1)
            obs[i, 1] = i / max(1, self.n - 1)
            obs[i, 2] = min(len(self.row['query'].split()), 30) / 30
            obs[i, 3] = len(self.selected) / 4
            obs[i, 4] = self.searches / 4
            obs[i, 5] = float(self.done_agent[i])
            if self.variants[i] >= 0:
                obs[i, 6 + self.variants[i]] = 1
            if self.communication and self.n > 1:
                j = 1 - i
                obs[i, 9] = self.last_actions[j] / 5
                obs[i, 10] = float(self.variants[j] == self.variants[i] and self.variants[i] >= 0)
            if not self.done_agent[i]:
                if self.t % 2 == 0:
                    mask[i, 1:4] = True
                else:
                    hits = self.hits(i)[:5]
                    mask[i, 1:len(hits) + 1] = True
                    for k, hit in enumerate(hits):
                        obs[i, 11 + k] = np.tanh(float(hit['score']) / 20)
                        obs[i, 16 + k] = float(hit['id'] in self.selected)
            if self.t % 2 == 0:
                obs[i, 11:16] = 0
        return obs, mask

    def step(self, actions):
        _, mask = self.observe()
        self.mask_checks += len(actions)
        if len(actions) != self.n or any(int(a) != a or a < 0 or a >= self.ACTIONS or not mask[i, int(a)] for i, a in enumerate(actions)):
            raise ValueError('Action violates the current action mask')
        reward = 0.0
        if self.t % 2 == 0:
            choices = []
            for i, action in enumerate(actions):
                if self.done_agent[i]:
                    continue
                if action == 0:
                    self.done_agent[i] = True
                else:
                    self.variants[i] = int(action) - 1
                    choices.append(int(action))
                    self.searches += 1
                    reward -= self.cost
            self.duplicate_searches += len(choices) - len(set(choices))
        else:
            for i, action in enumerate(actions):
                if self.done_agent[i]:
                    continue
                if action == 0:
                    self.done_agent[i] = True
                    continue
                hits = self.hits(i)
                if action > len(hits):
                    raise ValueError('Invalid evidence action')
                docid = hits[int(action) - 1]['id']
                if docid not in self.selected:
                    self.selected.append(docid)
                    reward += float(self.judgments.get(docid, 0) > 0)
        self.last_actions = list(map(int, actions))
        self.t += 1
        done = self.t == 2 * self.rounds
        return reward, done

    def metrics(self):
        positives = {d for d, rel in self.judgments.items() if rel > 0}
        found = sum(d in positives for d in self.selected)
        return {'recall': found / len(positives) if positives else 0.0,
                'hit': float(found > 0), 'precision': found / max(1, len(self.selected)),
                'searches': self.searches, 'selected': len(self.selected),
                'duplicate_searches': self.duplicate_searches,
                'mask_checks': self.mask_checks, 'invalid_actions': 0}
