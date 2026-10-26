"""Tests : python -m unittest discover -s tests"""
import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import config as C, mdp
from app.agents import QLearning, DoubleQLearning, RoundRobin
from app.episode import run_episode
from app.simulator import LoadBalancerEnv


class Core(unittest.TestCase):
    def test_states(self):
        self.assertEqual(C.N_STATES, 27)
        self.assertEqual(mdp.encode_state([0, 2, 1]), 7)
        self.assertEqual(mdp.decode_state(7), [0, 2, 1])
        self.assertEqual([mdp.cpu_level(x) for x in (10, 39.9, 40, 70, 70.1)], [0, 0, 1, 1, 2])

    def test_reward_decreases_with_cost(self):
        good, _ = mdp.compute_reward(40, 30, False, False)
        bad, _ = mdp.compute_reward(80, 50, False, False)
        fail, _ = mdp.compute_reward(500, 100, True, False)
        self.assertGreater(good, bad)
        self.assertGreater(bad, fail)

    def test_same_seed_same_traffic(self):
        a = LoadBalancerEnv(0.7, seed=5); b = LoadBalancerEnv(0.7, seed=5)
        ra = run_episode(a, RoundRobin()); rb = run_episode(b, RoundRobin())
        self.assertEqual(ra["reward"], rb["reward"])

    def test_double_q_example_from_document(self):
        # section 4.2 : QA choisit le serveur A (8), QB l'évalue à 4 -> valeur future utilisée = 4
        ag = DoubleQLearning(lr=1.0, gamma=1.0, seed=0)
        ag.QA[1] = [8, 6, 5]; ag.QB[1] = [4, 7, 3]
        ag.rng.random = lambda: 0.0        # force la mise à jour de QA
        log = ag.update(0, 0, 0.0, 1, False, [0, 1, 2])
        self.assertEqual(log["table"], "QA")
        self.assertEqual(log["future"], 4)

    def test_q_update_formula(self):
        ag = QLearning(lr=0.5, gamma=0.9)
        ag.Q[0][0] = 2.0; ag.Q[1] = [1.0, 5.0, 3.0]
        log = ag.update(0, 0, 1.0, 1, False, [0, 1, 2])
        self.assertAlmostEqual(log["target"], 1.0 + 0.9 * 5.0)
        self.assertAlmostEqual(ag.Q[0][0], 2.0 + 0.5 * (5.5 - 2.0))

    def test_down_server_masked(self):
        env = LoadBalancerEnv(0.7, seed=1); env.inject_fault(0)
        self.assertNotIn(0, env.valid_actions())


if __name__ == "__main__":
    unittest.main()
