import unittest
import numpy as np
from environment import SearchEnvironment


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.row = {'qid': 'q', 'query': 'Тестовый запрос', 'variants': [
            {'hits': [{'id': str(j) + str(i), 'score': 5 - i} for i in range(5)]}
            for j in range(3)]}

    def test_qrels_never_change_observations(self):
        a = SearchEnvironment(self.row, {'00': 1})
        b = SearchEnvironment(self.row, {'10': 1})
        for actions in ([1, 2], [1, 1], [2, 3], [1, 1]):
            np.testing.assert_array_equal(a.observe()[0], b.observe()[0])
            np.testing.assert_array_equal(a.observe()[1], b.observe()[1])
            a.step(actions); b.step(actions)

    def test_budget_and_deduplication(self):
        e = SearchEnvironment(self.row, {'00': 1})
        for actions in ([1, 1], [1, 1], [1, 1], [1, 1]):
            reward, done = e.step(actions)
        self.assertTrue(done)
        self.assertEqual(e.searches, 4)
        self.assertEqual(e.selected, ['00'])
        self.assertEqual(e.metrics()['recall'], 1)

    def test_invalid_actions_raise(self):
        e = SearchEnvironment(self.row, {})
        with self.assertRaises(ValueError):
            e.step([5, 1])

    def test_single_agent_same_search_limit(self):
        e = SearchEnvironment(self.row, {}, n_agents=1)
        for _ in range(4):
            e.step([1]); _, done = e.step([1])
        self.assertTrue(done)
        self.assertEqual(e.searches, 4)


if __name__ == '__main__':
    unittest.main()
